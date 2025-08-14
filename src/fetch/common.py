import os
import csv
import time
import math
import queue
import threading
from dataclasses import dataclass
from typing import Optional, Iterable, List, Dict, Tuple

import requests
from astropy.io import fits


MANIFEST_HEADERS = [
	'remote_url',
	'local_path',
	'status',
	'http_status',
	'bytes',
]


@dataclass
class DownloadResult:
	remote_url: str
	local_path: str
	status: str
	http_status: int
	bytes: int
	message: Optional[str] = None


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def write_manifest(rows: Iterable[Dict[str, object]], out_csv: str) -> None:
	ensure_dir(os.path.dirname(out_csv))
	with open(out_csv, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=MANIFEST_HEADERS)
		w.writeheader()
		for r in rows:
			w.writerow({k: r.get(k) for k in MANIFEST_HEADERS})


def append_manifest(results: Iterable[DownloadResult], out_csv: str) -> None:
	ensure_dir(os.path.dirname(out_csv))
	file_exists = os.path.exists(out_csv)
	with open(out_csv, 'a', newline='') as f:
		w = csv.DictWriter(f, fieldnames=MANIFEST_HEADERS)
		if not file_exists:
			w.writeheader()
		for r in results:
			w.writerow({
				'remote_url': r.remote_url,
				'local_path': r.local_path,
				'status': r.status,
				'http_status': r.http_status,
				'bytes': r.bytes,
			})


def log_failures(results: Iterable[DownloadResult], log_path: str) -> None:
	"""Append human-readable failure reasons to a log file."""
	if not results:
		return
	ensure_dir(os.path.dirname(log_path))
	with open(log_path, 'a') as f:
		for r in results:
			if r.status != 'ok':
				f.write(f"{r.status}\t{r.http_status}\t{r.bytes}\t{r.remote_url}\t{r.local_path}\n")


def http_head(url: str, timeout: int = 30) -> Tuple[int, int]:
	try:
		resp = requests.head(url, timeout=timeout, allow_redirects=True)
		return resp.status_code, int(resp.headers.get('Content-Length', '0') or 0)
	except Exception:
		return 0, 0


def stream_download(url: str, dest_path: str, timeout: int = 60, chunk_size: int = 1 << 16) -> DownloadResult:
	ensure_dir(os.path.dirname(dest_path))
	# Support resume
	existing = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
	head_status, content_len = http_head(url, timeout=timeout)
	headers = {}
	mode = 'wb'
	if existing > 0 and content_len and existing < content_len:
		headers['Range'] = f'bytes={existing}-'
		mode = 'ab'
	try:
		with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
			r.raise_for_status()
			with open(dest_path, mode) as f:
				for chunk in r.iter_content(chunk_size=chunk_size):
					if not chunk:
						continue
					f.write(chunk)
		final_size = os.path.getsize(dest_path)
		return DownloadResult(url, dest_path, 'ok', head_status or 200, final_size)
	except Exception as e:
		return DownloadResult(url, dest_path, 'error', getattr(e, 'response', None).status_code if hasattr(e, 'response') and e.response else 0, os.path.getsize(dest_path) if os.path.exists(dest_path) else 0, str(e))


def verify_fits_basic(path: str, required_headers: Optional[List[str]] = None) -> bool:
	try:
		with fits.open(path, memmap=False) as hdul:
			_ = len(hdul)
			if required_headers:
				h = hdul[0].header
				for k in required_headers:
					if k not in h:
						return False
		return True
	except Exception:
		return False


def parallel_download(url_to_path: List[Tuple[str, str]], concurrency: int = 8, timeout: int = 60,
					  verify_cb=None) -> List[DownloadResult]:
	q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
	for u, p in url_to_path:
		q.put((u, p))
	results: List[DownloadResult] = []
	results_lock = threading.Lock()

	def worker() -> None:
		while True:
			try:
				u, p = q.get_nowait()
			except queue.Empty:
				return
			res = stream_download(u, p, timeout=timeout)
			if res.status == 'ok' and verify_cb is not None:
				try:
					if not verify_cb(p):
						res.status = 'bad_fits'
				except Exception:
					res.status = 'bad_fits'
			with results_lock:
				results.append(res)
			q.task_done()

	threads: List[threading.Thread] = []
	for i in range(max(1, concurrency)):
		t = threading.Thread(target=worker, daemon=True)
		t.start()
		threads.append(t)
	for t in threads:
		t.join()
	return results
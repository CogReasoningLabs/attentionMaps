"""Lazy, local fastText fallback for records without metadata language evidence."""

import hashlib
import importlib.metadata
import math
import os
import sys
from pathlib import Path
import tempfile
import urllib.request

MODEL_ROOT = Path(__file__).resolve().parents[2] / 'artifacts/language_models'
DETECTION_DEFAULTS = {
    'language_detection': 'none',
    'language_detection_model': str(MODEL_ROOT / 'lid.176.bin'),
    'language_detection_min_confidence': .80,
    'language_detection_min_letters': 20,
    'language_detection_max_characters': 4000,
}
MODEL_URLS = {name: 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/' + name
              for name in ('lid.176.ftz', 'lid.176.bin')}


def detection_settings(settings=None):
    settings = settings or {}
    values = {key: settings.get(key, value) for key, value in DETECTION_DEFAULTS.items()}
    if values['language_detection'] not in ('none', 'fasttext'):
        raise ValueError('language_detection must be none or fasttext')
    path = values['language_detection_model']
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError('language_detection_model must be a non-empty .bin or .ftz path')
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in ('.bin', '.ftz'):
        raise ValueError('language_detection_model must be a .bin or .ftz path')
    values['language_detection_model'] = str(path)
    score = values['language_detection_min_confidence']
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('language_detection_min_confidence must be between 0 and 1')
    for key in ('language_detection_min_letters', 'language_detection_max_characters'):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if values['language_detection_max_characters'] < values['language_detection_min_letters']:
        raise ValueError('language_detection_max_characters must be at least language_detection_min_letters')
    return values


def _download_model(path):
    url = MODEL_URLS.get(path.name)
    if not url:
        raise ValueError(f'Language model does not exist: {path}. Supply a local fastText model or use lid.176.ftz/lid.176.bin.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size < 1024:
            raise ValueError('Downloaded language model is incomplete')
        try:
            os.link(temporary, path)  # Never overwrite a user-supplied or concurrently downloaded model.
        except FileExistsError:
            pass
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


class FastTextLanguageDetector:
    def __init__(self, settings):
        self.settings = detection_settings(settings)
        self.path = Path(self.settings['language_detection_model'])
        self._model = None
        self.metadata = {
            'method': 'fasttext', 'model_path': str(self.path), 'model_sha256': None,
            'model_url': MODEL_URLS.get(self.path.name),
            'min_confidence': self.settings['language_detection_min_confidence'],
            'min_letters': self.settings['language_detection_min_letters'],
            'max_characters': self.settings['language_detection_max_characters'],
            'unit': 'one dominant-language prediction per record',
            'note': 'Scores are model outputs, not calibrated accuracy. Short, romanized and code-switched text can be unreliable. Model input may be truncated; script analysis uses the full selected text.',
        }

    def ensure_model_file(self):
        """Prepare one shared model file before multiple workers start."""
        if not self.path.exists():
            if self.path.name in MODEL_URLS:
                print(f'Downloading language detector to {self.path}', file=sys.stderr)
            _download_model(self.path)

    def _load(self):
        if self._model is not None:
            return
        try:
            import fasttext
        except ImportError as error:
            raise ImportError('Text language detection requires fasttext: venv/bin/python -m pip install "fasttext>=0.9.3,<0.10"') from error
        self.ensure_model_file()
        try:
            self._model = fasttext.load_model(str(self.path))
        except (ValueError, RuntimeError) as error:
            raise ValueError(f'Could not load fastText language model {self.path}: {error}') from error
        with self.path.open('rb') as stream:
            self.metadata['model_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        try:
            self.metadata['package_version'] = importlib.metadata.version('fasttext')
        except importlib.metadata.PackageNotFoundError:
            self.metadata['package_version'] = 'unknown'

    def detect_many(self, texts):
        """Predict several independent records with one fastText batch call."""
        limit = self.settings['language_detection_max_characters']
        min_letters = self.settings['language_detection_min_letters']
        results = []
        prepared = []
        prepared_indexes = []
        for text in texts:
            portion = text[:limit]
            cleaned = ' '.join(portion.replace('\x00', ' ').split())
            result = {'language': None, 'score': None, 'characters': len(portion),
                      'truncated': len(text) > limit, 'reason': 'too_short'}
            results.append(result)
            # We only need to know whether the threshold is reached. Most
            # records have enough letters near the start; stop counting then.
            letters = 0
            for character in cleaned:
                if character.isalpha():
                    letters += 1
                    if letters >= min_letters:
                        break
            if letters >= min_letters:
                prepared_indexes.append(len(results) - 1)
                prepared.append(cleaned)
        if not prepared:
            return results

        self._load()
        # Always use fastText's public list API (including one-item batches);
        # it avoids the NumPy 2 scalar copy=False compatibility issue.
        try:
            labels, scores = self._model.predict(prepared, k=1)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f'fastText language prediction failed: {error}') from error
        for position, result_index in enumerate(prepared_indexes):
            result = results[result_index]
            if (position >= len(labels) or position >= len(scores)
                    or not len(labels[position]) or not len(scores[position])):
                result['reason'] = 'no_prediction'
                continue
            label, score = str(labels[position][0]), float(scores[position][0])
            if not label.startswith('__label__') or not label.removeprefix('__label__') or not math.isfinite(score):
                result['reason'] = 'invalid_prediction'
                continue
            score = min(1.0, max(0.0, score))
            result.update(score=score, reason='low_confidence')
            if score >= self.settings['language_detection_min_confidence']:
                result.update(language=label.removeprefix('__label__'), reason='accepted')
        return results

    def detect(self, text):
        return self.detect_many([text])[0]


def make_language_detector(settings=None):
    values = detection_settings(settings)
    return FastTextLanguageDetector(values) if values['language_detection'] == 'fasttext' else None

import librosa
import numpy as np
import logging
from typing import Tuple, Optional, List
from .harmonic_analyzer import HarmonicAnalyzer

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.harmonic_analyzer = HarmonicAnalyzer()
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def detect_bpm(self, y: np.ndarray, sr: float) -> Tuple[float, np.ndarray]:
        try:
            tempo, beats = librosa.beat.beat_track(
                y=y, 
                sr=sr, 
                hop_length=self.hop_length
            )
            
            if tempo < 60 or tempo > 200:
                logger.warning(f"Detected BPM {tempo:.1f} is outside typical range (60-200)")
                tempo = self._fallback_bpm_detection(y, sr)
                
            return float(tempo), beats
        except Exception as e:
            logger.error(f"Error detecting BPM: {e}")
            return self._fallback_bpm_detection(y, sr)
    
    def _fallback_bpm_detection(self, y: np.ndarray, sr: float) -> Tuple[float, np.ndarray]:
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
            tempo = librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr))[0]
            
            if tempo < 60:
                tempo *= 2
            elif tempo > 200:
                tempo /= 2
                
            beats = librosa.frames_to_samples(onset_frames, hop_length=self.hop_length)
            return float(tempo), beats
        except Exception as e:
            logger.error(f"Fallback BPM detection failed: {e}")
            return 120.0, np.array([])
    
    def get_beat_positions(self, y: np.ndarray, sr: float, beats: np.ndarray) -> np.ndarray:
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        return beat_times
    
    def analyze_energy(self, y: np.ndarray, sr: float, window_size: int = 2048) -> np.ndarray:
        hop_length = window_size // 4
        energy = []
        
        for i in range(0, len(y) - window_size, hop_length):
            window = y[i:i + window_size]
            rms_energy = np.sqrt(np.mean(window ** 2))
            energy.append(rms_energy)
            
        return np.array(energy)
    
    def find_mix_points(self, y: np.ndarray, sr: float, beats: np.ndarray, 
                       start_ratio: float = 0.75, end_ratio: float = 0.25) -> Tuple[float, float]:
        duration = len(y) / sr
        
        start_time = duration * start_ratio
        end_time = duration * end_ratio
        
        beat_times = self.get_beat_positions(y, sr, beats)
        
        if len(beat_times) > 0:
            start_beat_idx = np.argmin(np.abs(beat_times - start_time))
            start_time = beat_times[start_beat_idx]
            
            end_beat_idx = np.argmin(np.abs(beat_times - end_time))
            end_time = beat_times[end_beat_idx]
        
        return start_time, end_time
    
    def detect_key(self, y: np.ndarray, sr: float) -> dict:
        try:
            key_info = self.harmonic_analyzer.detect_key_detailed(y, sr)
            return key_info
        except Exception as e:
            logger.error(f"Error detecting key: {e}")
            return {'key': 'Unknown', 'camelot': 'Unknown', 'confidence': 0.0}
    
    def analyze_track(self, file_path: str) -> dict:
        try:
            y, sr = self.load_audio(file_path)
            tempo, beats = self.detect_bpm(y, sr)
            beat_times = self.get_beat_positions(y, sr, beats)
            energy = self.analyze_energy(y, sr)
            key_info = self.detect_key(y, sr)
            
            mix_start, mix_end = self.find_mix_points(y, sr, beats)
            
            spectral_features = self._extract_spectral_features(y, sr)
            
            analysis = {
                'file_path': file_path,
                'tempo': tempo,
                'beats': beats,
                'beat_times': beat_times,
                'energy': energy,
                'key': key_info.get('key', 'Unknown'),
                'camelot': key_info.get('camelot', 'Unknown'),
                'key_confidence': key_info.get('confidence', 0.0),
                'duration': len(y) / sr,
                'mix_start_time': mix_start,
                'mix_end_time': mix_end,
                'sample_rate': sr,
                'spectral_features': spectral_features,
                'audio_data': y
            }
            
            logger.info(f"Analyzed {file_path}: BPM={tempo:.1f}, Key={analysis['key']}, Duration={analysis['duration']:.1f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze track {file_path}: {e}")
            raise
    
    def _extract_spectral_features(self, y: np.ndarray, sr: float) -> dict:
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'mfcc_means': [float(np.mean(mfcc)) for mfcc in mfccs],
                'brightness': float(np.mean(spectral_centroids) / (sr / 2))
            }
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return {}
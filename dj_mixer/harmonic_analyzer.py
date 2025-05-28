import librosa
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)

class HarmonicAnalyzer:
    
    CAMELOT_WHEEL = {
        'C': '8B', 'Db': '3B', 'D': '10B', 'Eb': '5B', 'E': '12B', 'F': '7B',
        'F#': '2B', 'G': '9B', 'Ab': '4B', 'A': '11B', 'Bb': '6B', 'B': '1B',
        'Cm': '5A', 'C#m': '12A', 'Dm': '7A', 'Ebm': '2A', 'Em': '9A', 'Fm': '4A',
        'F#m': '11A', 'Gm': '6A', 'G#m': '1A', 'Am': '8A', 'Bbm': '3A', 'Bm': '10A'
    }
    
    KEY_PROFILES = {
        'major': np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
        'minor': np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    }
    
    @classmethod
    def detect_key_detailed(cls, y: np.ndarray, sr: float) -> Dict[str, str]:
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
            chroma_mean = np.mean(chroma, axis=1)
            
            chroma_mean = chroma_mean / np.sum(chroma_mean)
            
            key_scores = {}
            
            for shift in range(12):
                shifted_chroma = np.roll(chroma_mean, shift)
                
                major_score = np.corrcoef(shifted_chroma, cls.KEY_PROFILES['major'])[0, 1]
                minor_score = np.corrcoef(shifted_chroma, cls.KEY_PROFILES['minor'])[0, 1]
                
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key_name = key_names[shift]
                
                key_scores[f"{key_name}"] = major_score
                key_scores[f"{key_name}m"] = minor_score
            
            best_key = max(key_scores, key=key_scores.get)
            confidence = key_scores[best_key]
            
            normalized_key = cls._normalize_key_name(best_key)
            camelot = cls.CAMELOT_WHEEL.get(normalized_key, 'Unknown')
            
            return {
                'key': normalized_key,
                'camelot': camelot,
                'confidence': float(confidence),
                'raw_scores': key_scores
            }
            
        except Exception as e:
            logger.error(f"Error in detailed key detection: {e}")
            return {'key': 'Unknown', 'camelot': 'Unknown', 'confidence': 0.0, 'raw_scores': {}}
    
    @classmethod
    def _normalize_key_name(cls, key: str) -> str:
        key_mapping = {
            'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb',
            'C#m': 'Dbm', 'D#m': 'Ebm', 'F#m': 'Gbm', 'G#m': 'Abm', 'A#m': 'Bbm'
        }
        return key_mapping.get(key, key)
    
    @classmethod
    def calculate_key_compatibility(cls, key1: str, key2: str) -> Dict[str, float]:
        camelot1 = cls.CAMELOT_WHEEL.get(key1, 'Unknown')
        camelot2 = cls.CAMELOT_WHEEL.get(key2, 'Unknown')
        
        if camelot1 == 'Unknown' or camelot2 == 'Unknown':
            return {'compatibility': 0.0, 'rule': 'unknown_key'}
        
        num1, letter1 = camelot1[:-1], camelot1[-1]
        num2, letter2 = camelot2[:-1], camelot2[-1]
        
        try:
            num1, num2 = int(num1), int(num2)
        except ValueError:
            return {'compatibility': 0.0, 'rule': 'invalid_camelot'}
        
        if camelot1 == camelot2:
            return {'compatibility': 1.0, 'rule': 'perfect_match'}
        
        if letter1 == letter2:
            diff = min(abs(num1 - num2), 12 - abs(num1 - num2))
            if diff == 1:
                return {'compatibility': 0.9, 'rule': 'adjacent_same_mode'}
            elif diff == 7:
                return {'compatibility': 0.7, 'rule': 'fifth_circle'}
        
        if abs(num1 - num2) == 0 and letter1 != letter2:
            return {'compatibility': 0.8, 'rule': 'relative_major_minor'}
        
        if letter1 != letter2:
            diff = min(abs(num1 - num2), 12 - abs(num1 - num2))
            if diff == 1:
                return {'compatibility': 0.6, 'rule': 'adjacent_different_mode'}
        
        return {'compatibility': 0.2, 'rule': 'poor_match'}
    
    @classmethod
    def get_compatible_keys(cls, key: str, min_compatibility: float = 0.6) -> List[Dict[str, str]]:
        compatible = []
        
        for other_key, camelot in cls.CAMELOT_WHEEL.items():
            if other_key == key:
                continue
                
            compatibility = cls.calculate_key_compatibility(key, other_key)
            if compatibility['compatibility'] >= min_compatibility:
                compatible.append({
                    'key': other_key,
                    'camelot': camelot,
                    'compatibility': compatibility['compatibility'],
                    'rule': compatibility['rule']
                })
        
        return sorted(compatible, key=lambda x: x['compatibility'], reverse=True)
    
    @classmethod
    def analyze_harmonic_flow(cls, track_keys: List[str]) -> Dict[str, float]:
        if len(track_keys) < 2:
            return {'average_compatibility': 1.0, 'flow_score': 1.0}
        
        compatibilities = []
        
        for i in range(len(track_keys) - 1):
            comp = cls.calculate_key_compatibility(track_keys[i], track_keys[i + 1])
            compatibilities.append(comp['compatibility'])
        
        avg_compatibility = np.mean(compatibilities)
        
        flow_penalties = []
        for i in range(len(compatibilities) - 1):
            if compatibilities[i] > 0.8 and compatibilities[i + 1] < 0.4:
                flow_penalties.append(0.3)
            elif compatibilities[i] < 0.4 and compatibilities[i + 1] > 0.8:
                flow_penalties.append(0.1)
        
        flow_penalty = np.mean(flow_penalties) if flow_penalties else 0
        flow_score = max(0, avg_compatibility - flow_penalty)
        
        return {
            'average_compatibility': float(avg_compatibility),
            'flow_score': float(flow_score),
            'transition_scores': compatibilities
        }
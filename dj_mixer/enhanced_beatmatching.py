#!/usr/bin/env python3

import numpy as np
import librosa
from pydub import AudioSegment
from typing import Dict, List, Tuple, Optional
from .harmonic_analyzer import HarmonicAnalyzer
import logging

logger = logging.getLogger(__name__)

class EnhancedBeatmatching:
    """
    Enhanced beatmatching system with natural tempo relationships and harmonic mixing
    """
    
    def __init__(self):
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.tempo_cache = {}
        
    def analyze_comprehensive_compatibility(self, track1_analysis: Dict, 
                                          track2_analysis: Dict) -> Dict:
        """Comprehensive compatibility analysis including BPM, key, and energy"""
        logger.debug("Analyzing comprehensive track compatibility")
        
        # BPM compatibility
        bpm_compat = self._analyze_bpm_compatibility(
            track1_analysis.get('tempo', track1_analysis.get('bpm', 120)), 
            track2_analysis.get('tempo', track2_analysis.get('bpm', 120))
        )
        
        # Key compatibility
        key_compat = self.harmonic_analyzer.calculate_key_compatibility(
            track1_analysis.get('key', 'Unknown'),
            track2_analysis.get('key', 'Unknown')
        )
        
        # Energy compatibility
        energy_compat = self._analyze_energy_compatibility(
            track1_analysis.get('energy', 0.5),
            track2_analysis.get('energy', 0.5)
        )
        
        # Overall compatibility score
        weights = {'bpm': 0.4, 'key': 0.4, 'energy': 0.2}
        overall_score = (
            bpm_compat['compatibility'] * weights['bpm'] +
            key_compat['compatibility'] * weights['key'] +
            energy_compat['compatibility'] * weights['energy']
        )
        
        return {
            'overall_compatibility': overall_score,
            'bpm_analysis': bpm_compat,
            'key_analysis': key_compat,
            'energy_analysis': energy_compat,
            'mixing_recommendation': self._get_mixing_recommendation(
                bpm_compat, key_compat, energy_compat
            )
        }
    
    def _analyze_bpm_compatibility(self, bpm1: float, bpm2: float) -> Dict:
        """Analyze BPM compatibility with natural tempo relationships"""
        bpm_ratio = bpm2 / bpm1
        
        # Define natural tempo relationships
        natural_ratios = {
            1.0: {'name': 'perfect_match', 'compatibility': 1.0},
            0.5: {'name': 'half_time', 'compatibility': 0.95},
            2.0: {'name': 'double_time', 'compatibility': 0.95},
            0.75: {'name': 'three_quarter_time', 'compatibility': 0.9},
            1.33: {'name': 'four_third_time', 'compatibility': 0.9},
            0.67: {'name': 'two_third_time', 'compatibility': 0.85},
            1.5: {'name': 'three_half_time', 'compatibility': 0.85},
        }
        
        # Find closest natural ratio
        closest_ratio = min(natural_ratios.keys(), key=lambda x: abs(x - bpm_ratio))
        ratio_diff = abs(closest_ratio - bpm_ratio)
        
        if ratio_diff < 0.05:  # Within 5% of natural ratio
            relationship = natural_ratios[closest_ratio]
            compatibility = relationship['compatibility']
        else:
            # Calculate compatibility based on how far from natural ratios
            min_stretch = min(abs(bpm_ratio - ratio) for ratio in natural_ratios.keys())
            if min_stretch < 0.1:  # Within 10%
                compatibility = 0.8 - (min_stretch * 5)  # Penalty for stretching
            elif min_stretch < 0.2:  # Within 20%
                compatibility = 0.6 - (min_stretch * 2)
            else:
                compatibility = max(0.1, 0.5 - min_stretch)
            
            relationship = {'name': 'requires_stretching', 'compatibility': compatibility}
        
        return {
            'bpm1': bpm1,
            'bpm2': bpm2,
            'ratio': bpm_ratio,
            'closest_natural_ratio': closest_ratio,
            'relationship': relationship,
            'compatibility': compatibility,
            'stretch_amount': abs(bpm_ratio - closest_ratio)
        }
    
    def _analyze_energy_compatibility(self, energy1, energy2) -> Dict:
        """Analyze energy level compatibility"""
        # Handle both arrays and scalar values
        if isinstance(energy1, np.ndarray):
            energy1 = np.mean(energy1)
        if isinstance(energy2, np.ndarray):
            energy2 = np.mean(energy2)
            
        energy_diff = abs(energy1 - energy2)
        
        if energy_diff < 0.1:
            flow_type = 'maintain_energy'
            compatibility = 1.0
        elif energy2 > energy1:
            energy_gain = energy2 - energy1
            if energy_gain > 0.3:
                flow_type = 'big_energy_boost'
                compatibility = 0.9  # Can be great for builds
            else:
                flow_type = 'energy_boost'
                compatibility = 0.95
        else:
            energy_drop = energy1 - energy2
            if energy_drop > 0.3:
                flow_type = 'big_energy_drop'
                compatibility = 0.7  # More challenging transition
            else:
                flow_type = 'energy_drop'
                compatibility = 0.8
        
        return {
            'energy1': energy1,
            'energy2': energy2,
            'energy_change': energy2 - energy1,
            'flow_type': flow_type,
            'compatibility': compatibility
        }
    
    def _get_mixing_recommendation(self, bpm_analysis: Dict, key_analysis: Dict, 
                                 energy_analysis: Dict) -> Dict:
        """Get specific mixing recommendations based on analysis"""
        recommendations = {
            'transition_style': 'crossfade',
            'eq_strategy': 'standard',
            'timing_advice': 'phrase_aligned',
            'special_techniques': []
        }
        
        # BPM-based recommendations
        if bpm_analysis['relationship']['name'] == 'perfect_match':
            recommendations['timing_advice'] = 'beat_aligned'
        elif bpm_analysis['relationship']['name'] in ['half_time', 'double_time']:
            recommendations['special_techniques'].append('tempo_play')
        elif bpm_analysis['stretch_amount'] > 0.1:
            recommendations['special_techniques'].append('gradual_tempo_adjustment')
        
        # Key-based recommendations
        if key_analysis['rule'] == 'perfect_match':
            recommendations['eq_strategy'] = 'minimal_eq'
        elif key_analysis['rule'] in ['adjacent_same_mode', 'relative_major_minor']:
            recommendations['eq_strategy'] = 'harmonic_blend'
        elif key_analysis['compatibility'] < 0.6:
            recommendations['special_techniques'].append('key_transition_effect')
        
        # Energy-based recommendations
        if energy_analysis['flow_type'] == 'big_energy_boost':
            recommendations['transition_style'] = 'build_up'
            recommendations['special_techniques'].append('energy_riser')
        elif energy_analysis['flow_type'] == 'big_energy_drop':
            recommendations['transition_style'] = 'breakdown'
            recommendations['special_techniques'].append('filter_sweep')
        
        return recommendations
    
    def create_perfect_beatmatch(self, track1: AudioSegment, track2: AudioSegment,
                               bpm1: float, bpm2: float, 
                               use_natural_relationships: bool = True) -> Tuple[AudioSegment, AudioSegment]:
        """Create perfect beatmatch between tracks"""
        logger.info(f"Creating beatmatch: {bpm1:.1f} BPM -> {bpm2:.1f} BPM")
        
        bpm_analysis = self._analyze_bpm_compatibility(bpm1, bpm2)
        
        if use_natural_relationships and bpm_analysis['stretch_amount'] < 0.05:
            # Use natural relationship - no stretching needed
            logger.debug(f"Using natural tempo relationship: {bpm_analysis['relationship']['name']}")
            return track1, track2
        
        # Calculate optimal tempo adjustment
        target_ratio = bpm_analysis['closest_natural_ratio']
        
        if abs(target_ratio - 1.0) < 0.02:  # Essentially no change needed
            return track1, track2
        
        # Apply intelligent tempo adjustment
        if target_ratio < 1.0:
            # Slow down track2 to match track1
            adjusted_track2 = self._apply_tempo_adjustment(track2, target_ratio)
            return track1, adjusted_track2
        else:
            # Speed up track2 or slow down track1
            if target_ratio < 1.2:  # Small speedup
                adjusted_track2 = self._apply_tempo_adjustment(track2, target_ratio)
                return track1, adjusted_track2
            else:  # Large change - adjust track1 instead
                adjusted_track1 = self._apply_tempo_adjustment(track1, 1.0 / target_ratio)
                return adjusted_track1, track2
    
    def _apply_tempo_adjustment(self, audio: AudioSegment, ratio: float) -> AudioSegment:
        """Apply tempo adjustment with minimal artifacts"""
        if abs(ratio - 1.0) < 0.01:
            return audio
        
        logger.debug(f"Applying tempo adjustment: ratio = {ratio:.3f}")
        
        # Convert to numpy for processing
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            # Process each channel separately
            processed_left = self._stretch_audio(samples[:, 0], ratio, audio.frame_rate)
            processed_right = self._stretch_audio(samples[:, 1], ratio, audio.frame_rate)
            processed_samples = np.column_stack((processed_left, processed_right))
            processed_samples = processed_samples.flatten()
        else:
            processed_samples = self._stretch_audio(samples, ratio, audio.frame_rate)
        
        # Convert back to AudioSegment
        processed_samples = np.clip(processed_samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(
            processed_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
    
    def _stretch_audio(self, samples: np.ndarray, ratio: float, sample_rate: int) -> np.ndarray:
        """Stretch audio using librosa's time stretching"""
        try:
            # Use librosa's time stretching for better quality
            stretched = librosa.effects.time_stretch(samples, rate=ratio)
            return stretched
        except Exception as e:
            logger.warning(f"Librosa time stretch failed: {e}, using simple interpolation")
            # Fallback to simple interpolation
            new_length = int(len(samples) / ratio)
            stretched = np.interp(
                np.linspace(0, len(samples) - 1, new_length),
                np.arange(len(samples)),
                samples
            )
            return stretched
    
    def optimize_track_order_by_compatibility(self, track_analyses: List[Dict]) -> List[int]:
        """Optimize track order for best harmonic and tempo flow"""
        logger.info("Optimizing track order for harmonic and tempo compatibility")
        
        if len(track_analyses) <= 2:
            return list(range(len(track_analyses)))
        
        # Create compatibility matrix
        n_tracks = len(track_analyses)
        compatibility_matrix = np.zeros((n_tracks, n_tracks))
        
        for i in range(n_tracks):
            for j in range(n_tracks):
                if i != j:
                    compat = self.analyze_comprehensive_compatibility(
                        track_analyses[i], track_analyses[j]
                    )
                    compatibility_matrix[i][j] = compat['overall_compatibility']
        
        # Use greedy algorithm to find good order
        order = self._greedy_order_optimization(compatibility_matrix)
        
        # Try to improve with local optimizations
        order = self._local_order_optimization(order, compatibility_matrix)
        
        return order
    
    def _greedy_order_optimization(self, compatibility_matrix: np.ndarray) -> List[int]:
        """Greedy algorithm for track ordering"""
        n_tracks = len(compatibility_matrix)
        used = set()
        order = []
        
        # Start with track that has best average compatibility
        avg_compat = np.mean(compatibility_matrix, axis=1)
        current = np.argmax(avg_compat)
        order.append(current)
        used.add(current)
        
        # Greedily add tracks with best compatibility to current track
        while len(order) < n_tracks:
            best_next = -1
            best_compat = -1
            
            for next_track in range(n_tracks):
                if next_track not in used:
                    compat = compatibility_matrix[current][next_track]
                    if compat > best_compat:
                        best_compat = compat
                        best_next = next_track
            
            if best_next != -1:
                order.append(best_next)
                used.add(best_next)
                current = best_next
            else:
                # Add remaining tracks
                remaining = [i for i in range(n_tracks) if i not in used]
                order.extend(remaining)
                break
        
        return order
    
    def _local_order_optimization(self, order: List[int], 
                                compatibility_matrix: np.ndarray) -> List[int]:
        """Local optimization using 2-opt swaps"""
        def calculate_order_score(ord_list):
            score = 0
            for i in range(len(ord_list) - 1):
                score += compatibility_matrix[ord_list[i]][ord_list[i + 1]]
            return score
        
        improved = True
        current_order = order.copy()
        current_score = calculate_order_score(current_order)
        
        # Try 2-opt swaps
        iterations = 0
        max_iterations = 20
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(1, len(current_order) - 1):
                for j in range(i + 1, len(current_order)):
                    # Try swapping positions i and j
                    new_order = current_order.copy()
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    
                    new_score = calculate_order_score(new_order)
                    
                    if new_score > current_score:
                        current_order = new_order
                        current_score = new_score
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_order
    
    def create_harmonic_journey(self, track_analyses: List[Dict], 
                              journey_style: str = 'energy_climb') -> List[int]:
        """Create harmonic journey with specific energy/mood progression"""
        logger.info(f"Creating {journey_style} harmonic journey")
        
        if journey_style == 'energy_climb':
            # Sort by energy, then optimize harmonically within energy groups
            return self._energy_climb_order(track_analyses)
        elif journey_style == 'harmonic_circle':
            # Follow the circle of fifths
            return self._harmonic_circle_order(track_analyses)
        elif journey_style == 'mood_journey':
            # Create emotional progression
            return self._mood_journey_order(track_analyses)
        else:
            # Default to compatibility optimization
            return self.optimize_track_order_by_compatibility(track_analyses)
    
    def _energy_climb_order(self, track_analyses: List[Dict]) -> List[int]:
        """Order tracks for gradual energy climb"""
        # Sort by energy (handle both arrays and scalar values)
        def get_energy_value(analysis):
            energy = analysis.get('energy', 0.5)
            if isinstance(energy, np.ndarray):
                return np.mean(energy)
            return energy
        
        energy_sorted = sorted(
            enumerate(track_analyses),
            key=lambda x: get_energy_value(x[1])
        )
        
        # Group into energy bands and optimize harmonically within each band
        n_bands = min(3, len(track_analyses))
        band_size = len(track_analyses) // n_bands
        
        final_order = []
        
        for band in range(n_bands):
            start_idx = band * band_size
            end_idx = start_idx + band_size if band < n_bands - 1 else len(track_analyses)
            
            band_tracks = [energy_sorted[i][0] for i in range(start_idx, end_idx)]
            band_analyses = [track_analyses[i] for i in band_tracks]
            
            # Optimize harmonically within this energy band
            if len(band_analyses) > 1:
                optimized_indices = self.optimize_track_order_by_compatibility(band_analyses)
                optimized_band = [band_tracks[i] for i in optimized_indices]
            else:
                optimized_band = band_tracks
            
            final_order.extend(optimized_band)
        
        return final_order
    
    def _harmonic_circle_order(self, track_analyses: List[Dict]) -> List[int]:
        """Order tracks following harmonic relationships"""
        # Group tracks by key
        key_groups = {}
        for i, analysis in enumerate(track_analyses):
            key = analysis.get('key', 'Unknown')
            if key not in key_groups:
                key_groups[key] = []
            key_groups[key].append(i)
        
        # Create harmonic flow between key groups
        ordered_keys = self._optimize_key_flow(list(key_groups.keys()))
        
        final_order = []
        for key in ordered_keys:
            final_order.extend(key_groups[key])
        
        return final_order
    
    def _optimize_key_flow(self, keys: List[str]) -> List[str]:
        """Optimize the flow between different keys"""
        if len(keys) <= 1:
            return keys
        
        # Create key compatibility matrix
        n_keys = len(keys)
        key_matrix = np.zeros((n_keys, n_keys))
        
        for i in range(n_keys):
            for j in range(n_keys):
                if i != j:
                    compat = self.harmonic_analyzer.calculate_key_compatibility(keys[i], keys[j])
                    key_matrix[i][j] = compat['compatibility']
        
        # Use greedy algorithm
        return [keys[i] for i in self._greedy_order_optimization(key_matrix)]
    
    def _mood_journey_order(self, track_analyses: List[Dict]) -> List[int]:
        """Create emotional journey through tracks"""
        # This is a simplified mood progression
        # In practice, you'd want more sophisticated mood analysis
        
        # For now, use energy + key characteristics to approximate mood
        mood_scores = []
        
        for i, analysis in enumerate(track_analyses):
            energy = analysis.get('energy', 0.5)
            key = analysis.get('key', 'C')
            
            # Simple mood scoring (major keys tend to be "happier")
            mood_score = energy
            if 'm' not in key:  # Major key
                mood_score += 0.2
            
            mood_scores.append((i, mood_score))
        
        # Sort by mood score and optimize harmonically
        mood_sorted = sorted(mood_scores, key=lambda x: x[1])
        mood_order = [x[0] for x in mood_sorted]
        
        # Apply harmonic optimization to the mood-sorted list
        mood_analyses = [track_analyses[i] for i in mood_order]
        harmonic_indices = self.optimize_track_order_by_compatibility(mood_analyses)
        
        return [mood_order[i] for i in harmonic_indices]
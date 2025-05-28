#!/usr/bin/env python3

import numpy as np
import librosa
from pydub import AudioSegment
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ProfessionalEQ:
    """
    Professional EQ system for proper frequency blending during transitions
    """
    
    def __init__(self):
        self.eq_cache = {}
        
    def analyze_frequency_spectrum(self, audio: AudioSegment) -> Dict:
        """Analyze frequency content of audio"""
        logger.debug("Analyzing frequency spectrum")
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = np.mean(samples, axis=1)  # Convert to mono for analysis
        
        # Normalize
        samples = samples / np.max(np.abs(samples))
        
        # Compute FFT
        fft = np.fft.rfft(samples)
        magnitude = np.abs(fft)
        
        # Define frequency bands
        sr = audio.frame_rate
        freqs = np.fft.rfftfreq(len(samples), 1/sr)
        
        # Split into frequency bands
        low_band = magnitude[(freqs >= 20) & (freqs <= 250)]      # Bass
        mid_low_band = magnitude[(freqs > 250) & (freqs <= 500)]  # Low mids
        mid_band = magnitude[(freqs > 500) & (freqs <= 2000)]     # Mids
        mid_high_band = magnitude[(freqs > 2000) & (freqs <= 4000)] # High mids
        high_band = magnitude[(freqs > 4000) & (freqs <= 20000)]  # Highs
        
        return {
            'low_energy': np.mean(low_band) if len(low_band) > 0 else 0,
            'mid_low_energy': np.mean(mid_low_band) if len(mid_low_band) > 0 else 0,
            'mid_energy': np.mean(mid_band) if len(mid_band) > 0 else 0,
            'mid_high_energy': np.mean(mid_high_band) if len(mid_high_band) > 0 else 0,
            'high_energy': np.mean(high_band) if len(high_band) > 0 else 0,
            'dominant_frequency': freqs[np.argmax(magnitude)],
            'frequency_balance': {
                'bass_heavy': np.mean(low_band) > np.mean(magnitude) * 1.5,
                'mid_heavy': np.mean(mid_band) > np.mean(magnitude) * 1.5,
                'bright': np.mean(high_band) > np.mean(magnitude) * 1.2
            }
        }
    
    def create_eq_transition_plan(self, track1_spectrum: Dict, track2_spectrum: Dict,
                                 transition_duration_ms: int, track1_analysis: Dict = None, 
                                 track2_analysis: Dict = None) -> Dict:
        """Create EQ transition plan to avoid frequency conflicts with musical timing"""
        logger.debug("Creating EQ transition plan")
        
        # Analyze potential conflicts
        conflicts = self._detect_frequency_conflicts(track1_spectrum, track2_spectrum)
        
        # Get musical timing information for better EQ transitions
        musical_timing = self._calculate_musical_eq_timing(
            transition_duration_ms, track1_analysis, track2_analysis
        )
        
        # Create EQ automation plan with musical timing
        eq_plan = {
            'track1_eq': self._plan_outgoing_eq(track1_spectrum, conflicts, transition_duration_ms, musical_timing),
            'track2_eq': self._plan_incoming_eq(track2_spectrum, conflicts, transition_duration_ms, musical_timing),
            'conflict_areas': conflicts,
            'musical_timing': musical_timing,
            'bassline_swap_point': musical_timing.get('bass_swap_point', transition_duration_ms // 2)
        }
        
        return eq_plan
    
    def _calculate_musical_eq_timing(self, duration_ms: int, track1_analysis: Dict = None, 
                                   track2_analysis: Dict = None) -> Dict:
        """Calculate musically appropriate timing for EQ changes"""
        timing = {
            'bass_swap_point': duration_ms // 2,  # Default fallback
            'early_cut_point': duration_ms // 4,
            'final_swap_point': duration_ms * 3 // 4
        }
        
        if track1_analysis:
            bpm1 = track1_analysis.get('tempo', track1_analysis.get('bpm', 120))
            beat_duration_ms = (60 / bpm1) * 1000
            
            # Align EQ changes to beat boundaries for musical coherence
            # Bass swap should happen on a strong beat (downbeat preferred)
            bars_in_transition = duration_ms / (beat_duration_ms * 4)
            
            if bars_in_transition >= 4:  # Long enough for proper phrasing
                # Early cut: Start of 2nd bar (after 1 bar)
                timing['early_cut_point'] = int(beat_duration_ms * 4)
                
                # Bass swap: Middle point, aligned to bar start
                mid_bar = int(bars_in_transition / 2)
                timing['bass_swap_point'] = int(mid_bar * beat_duration_ms * 4)
                
                # Final swap: Start of final bar
                final_bar = max(mid_bar + 1, int(bars_in_transition) - 1)
                timing['final_swap_point'] = int(final_bar * beat_duration_ms * 4)
            
            else:  # Short transition - align to beats
                # Align to nearest beats
                timing['early_cut_point'] = int(beat_duration_ms * 2)  # Beat 2
                timing['bass_swap_point'] = int(beat_duration_ms * 4)   # Beat 4
                timing['final_swap_point'] = int(beat_duration_ms * 6)  # Beat 6
        
        return timing
    
    def _detect_frequency_conflicts(self, spectrum1: Dict, spectrum2: Dict) -> Dict:
        """Detect potential frequency conflicts between tracks"""
        conflicts = {
            'bass_conflict': False,
            'mid_conflict': False,
            'high_conflict': False,
            'severity': 'low'
        }
        
        # Check bass conflict
        if (spectrum1['low_energy'] > 0.3 and spectrum2['low_energy'] > 0.3):
            conflicts['bass_conflict'] = True
        
        # Check mid conflict
        if (spectrum1['mid_energy'] > 0.4 and spectrum2['mid_energy'] > 0.4):
            conflicts['mid_conflict'] = True
        
        # Check high conflict
        if (spectrum1['high_energy'] > 0.3 and spectrum2['high_energy'] > 0.3):
            conflicts['high_conflict'] = True
        
        # Determine severity
        conflict_count = sum([conflicts['bass_conflict'], 
                            conflicts['mid_conflict'], 
                            conflicts['high_conflict']])
        
        if conflict_count >= 2:
            conflicts['severity'] = 'high'
        elif conflict_count == 1:
            conflicts['severity'] = 'medium'
        
        return conflicts
    
    def _plan_outgoing_eq(self, spectrum: Dict, conflicts: Dict, duration_ms: int, musical_timing: Dict = None) -> List[Dict]:
        """Plan EQ automation for outgoing track with musical timing"""
        eq_points = []
        
        # Use musical timing if available, otherwise fallback to percentage
        if musical_timing:
            early_cut = musical_timing['early_cut_point']
            bass_swap = musical_timing['bass_swap_point']
            final_point = musical_timing['final_swap_point']
        else:
            early_cut = duration_ms // 4
            bass_swap = duration_ms // 2
            final_point = duration_ms * 3 // 4
        
        # Start: Full spectrum
        eq_points.append({
            'time_ms': 0,
            'low_gain': 1.0,
            'mid_gain': 1.0,
            'high_gain': 1.0
        })
        
        # If bass conflict, start cutting bass early at musical moment
        if conflicts['bass_conflict']:
            eq_points.append({
                'time_ms': early_cut,  # Aligned to beat/bar
                'low_gain': 0.7,
                'mid_gain': 1.0,
                'high_gain': 1.0
            })
            
            # Bass swap point - cut bass completely at downbeat
            eq_points.append({
                'time_ms': bass_swap,  # Aligned to strong beat
                'low_gain': 0.0,
                'mid_gain': 1.0,
                'high_gain': 1.0
            })
        
        # If mid conflict, create space at final musical moment
        if conflicts['mid_conflict']:
            eq_points.append({
                'time_ms': final_point,  # Aligned to final phrase
                'low_gain': 0.0 if conflicts['bass_conflict'] else 0.8,
                'mid_gain': 0.6,
                'high_gain': 1.0
            })
        
        # End: Fade out completely
        eq_points.append({
            'time_ms': duration_ms,
            'low_gain': 0.0,
            'mid_gain': 0.0,
            'high_gain': 0.0
        })
        
        return eq_points
    
    def _plan_incoming_eq(self, spectrum: Dict, conflicts: Dict, duration_ms: int, musical_timing: Dict = None) -> List[Dict]:
        """Plan EQ automation for incoming track with musical timing"""
        eq_points = []
        
        # Use musical timing if available, otherwise fallback to percentage
        if musical_timing:
            early_cut = musical_timing['early_cut_point']
            bass_swap = musical_timing['bass_swap_point']
            final_point = musical_timing['final_swap_point']
        else:
            early_cut = duration_ms // 4
            bass_swap = duration_ms // 2
            final_point = duration_ms * 3 // 4
        
        # Start: No bass to avoid muddiness
        eq_points.append({
            'time_ms': 0,
            'low_gain': 0.0,
            'mid_gain': 0.8,
            'high_gain': 1.0
        })
        
        # Gradually introduce mids at early musical point
        if conflicts['mid_conflict']:
            eq_points.append({
                'time_ms': early_cut,  # Aligned to beat/bar
                'low_gain': 0.0,
                'mid_gain': 0.9,
                'high_gain': 1.0
            })
        
        # Bass introduction point - perfectly timed to downbeat
        if conflicts['bass_conflict']:
            eq_points.append({
                'time_ms': bass_swap,  # Aligned to strong beat
                'low_gain': 0.7,
                'mid_gain': 1.0,
                'high_gain': 1.0
            })
        
        # Full spectrum at final musical moment
        eq_points.append({
            'time_ms': final_point,  # Aligned to phrase boundary
            'low_gain': 1.0,
            'mid_gain': 1.0,
            'high_gain': 1.0
        })
        
        # End: Full spectrum
        eq_points.append({
            'time_ms': duration_ms,
            'low_gain': 1.0,
            'mid_gain': 1.0,
            'high_gain': 1.0
        })
        
        return eq_points
    
    def apply_eq_transition(self, track1: AudioSegment, track2: AudioSegment,
                          eq_plan: Dict, crossfade_duration_ms: int) -> Tuple[AudioSegment, AudioSegment]:
        """Apply EQ transition to both tracks"""
        logger.info("Applying professional EQ transition")
        
        # Extract transition sections
        track1_section = track1[-crossfade_duration_ms:]
        track2_section = track2[:crossfade_duration_ms]
        
        # Apply EQ automation to each track
        track1_eq = self._apply_eq_automation(track1_section, eq_plan['track1_eq'])
        track2_eq = self._apply_eq_automation(track2_section, eq_plan['track2_eq'])
        
        # Reconstruct full tracks
        track1_final = track1[:-crossfade_duration_ms] + track1_eq
        track2_final = track2_eq + track2[crossfade_duration_ms:]
        
        return track1_final, track2_final
    
    def _apply_eq_automation(self, audio: AudioSegment, eq_points: List[Dict]) -> AudioSegment:
        """Apply EQ automation curve to audio"""
        if not eq_points:
            return audio
        
        # Convert to numpy for processing
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        is_stereo = audio.channels == 2
        
        if is_stereo:
            samples = samples.reshape((-1, 2))
        
        duration_ms = len(audio)
        sample_rate = audio.frame_rate
        
        # Create EQ curves for each band
        low_curve = self._create_eq_curve(eq_points, 'low_gain', duration_ms)
        mid_curve = self._create_eq_curve(eq_points, 'mid_gain', duration_ms)
        high_curve = self._create_eq_curve(eq_points, 'high_gain', duration_ms)
        
        # Apply frequency band processing
        # Note: This is a simplified EQ implementation
        # In practice, you'd use proper IIR filters
        
        processed_samples = samples.copy()
        
        # Simple frequency domain processing
        if len(samples) > 1024:  # Only for longer segments
            # Apply per-frame processing
            frame_size = 1024
            hop_size = 512
            
            for i in range(0, len(samples) - frame_size, hop_size):
                frame_start = i
                frame_end = min(i + frame_size, len(samples))
                frame = samples[frame_start:frame_end]
                
                # Get EQ gains for this time point
                time_ms = (i / len(samples)) * duration_ms
                low_gain = np.interp(time_ms, [p['time_ms'] for p in eq_points], 
                                   [p['low_gain'] for p in eq_points])
                mid_gain = np.interp(time_ms, [p['time_ms'] for p in eq_points], 
                                   [p['mid_gain'] for p in eq_points])
                high_gain = np.interp(time_ms, [p['time_ms'] for p in eq_points], 
                                    [p['high_gain'] for p in eq_points])
                
                # Apply simple EQ (frequency domain)
                if is_stereo:
                    for ch in range(2):
                        processed_frame = self._apply_simple_eq(
                            frame[:, ch], sample_rate, low_gain, mid_gain, high_gain
                        )
                        processed_samples[frame_start:frame_end, ch] = processed_frame
                else:
                    processed_frame = self._apply_simple_eq(
                        frame, sample_rate, low_gain, mid_gain, high_gain
                    )
                    processed_samples[frame_start:frame_end] = processed_frame
        
        # Convert back to AudioSegment
        if is_stereo:
            processed_samples = processed_samples.flatten()
        
        processed_samples = np.clip(processed_samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(
            processed_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
    
    def _apply_simple_eq(self, frame: np.ndarray, sample_rate: int,
                        low_gain: float, mid_gain: float, high_gain: float) -> np.ndarray:
        """Apply simple 3-band EQ to audio frame"""
        # FFT-based EQ (simplified)
        fft = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(len(frame), 1/sample_rate)
        
        # Define frequency bands
        low_mask = freqs <= 250
        mid_mask = (freqs > 250) & (freqs <= 4000)
        high_mask = freqs > 4000
        
        # Apply gains
        fft[low_mask] *= low_gain
        fft[mid_mask] *= mid_gain
        fft[high_mask] *= high_gain
        
        # Convert back to time domain
        return np.fft.irfft(fft, len(frame))
    
    def _create_eq_curve(self, eq_points: List[Dict], param: str, duration_ms: int) -> np.ndarray:
        """Create smooth EQ automation curve"""
        if not eq_points:
            return np.ones(duration_ms)
        
        times = [p['time_ms'] for p in eq_points]
        values = [p[param] for p in eq_points]
        
        curve_times = np.linspace(0, duration_ms, duration_ms)
        curve = np.interp(curve_times, times, values)
        
        return curve
    
    def create_bassline_swap(self, track1: AudioSegment, track2: AudioSegment,
                           swap_point_ms: int, duration_ms: int = 2000) -> Tuple[AudioSegment, AudioSegment]:
        """Create smooth bassline swap between tracks"""
        logger.debug(f"Creating bassline swap at {swap_point_ms}ms")
        
        # Extract sections around swap point
        pre_swap = duration_ms // 2
        post_swap = duration_ms // 2
        
        track1_section = track1[swap_point_ms - pre_swap:swap_point_ms + post_swap]
        track2_section = track2[swap_point_ms - pre_swap:swap_point_ms + post_swap]
        
        # Create bass crossfade
        # Track1: Full bass -> No bass
        track1_bass_out = self._create_bass_fadeout(track1_section, pre_swap)
        
        # Track2: No bass -> Full bass
        track2_bass_in = self._create_bass_fadein(track2_section, pre_swap)
        
        # Reconstruct tracks
        track1_final = (track1[:swap_point_ms - pre_swap] + 
                       track1_bass_out + 
                       track1[swap_point_ms + post_swap:])
        
        track2_final = (track2[:swap_point_ms - pre_swap] + 
                       track2_bass_in + 
                       track2[swap_point_ms + post_swap:])
        
        return track1_final, track2_final
    
    def _create_bass_fadeout(self, audio: AudioSegment, fadeout_start_ms: int) -> AudioSegment:
        """Create bass fadeout for bassline swap"""
        # Simple implementation: gradually reduce low frequencies
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        is_stereo = audio.channels == 2
        
        if is_stereo:
            samples = samples.reshape((-1, 2))
        
        duration_ms = len(audio)
        fade_duration = duration_ms - fadeout_start_ms
        
        # Create bass gain curve
        bass_gain = np.ones(len(samples))
        fade_start_sample = int((fadeout_start_ms / duration_ms) * len(samples))
        
        for i in range(fade_start_sample, len(samples)):
            progress = (i - fade_start_sample) / (len(samples) - fade_start_sample)
            bass_gain[i] = 1.0 - progress  # Fade from 1.0 to 0.0
        
        # Apply bass filtering (simplified)
        processed_samples = samples.copy()
        
        # Simple bass reduction by attenuating low-frequency content
        # This is a simplified approach - proper implementation would use IIR filters
        if is_stereo:
            for ch in range(2):
                for i in range(len(samples)):
                    # Simple bass reduction (this is very basic)
                    processed_samples[i, ch] = samples[i, ch] * (0.3 + 0.7 * bass_gain[i])
        else:
            for i in range(len(samples)):
                processed_samples[i] = samples[i] * (0.3 + 0.7 * bass_gain[i])
        
        if is_stereo:
            processed_samples = processed_samples.flatten()
        
        processed_samples = np.clip(processed_samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(
            processed_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
    
    def _create_bass_fadein(self, audio: AudioSegment, fadein_end_ms: int) -> AudioSegment:
        """Create bass fadein for bassline swap"""
        # Start with no bass, gradually introduce it
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        is_stereo = audio.channels == 2
        
        if is_stereo:
            samples = samples.reshape((-1, 2))
        
        duration_ms = len(audio)
        fade_end_sample = int((fadein_end_ms / duration_ms) * len(samples))
        
        # Create bass gain curve
        bass_gain = np.ones(len(samples))
        
        for i in range(fade_end_sample):
            progress = i / fade_end_sample
            bass_gain[i] = progress  # Fade from 0.0 to 1.0
        
        # Apply bass introduction
        processed_samples = samples.copy()
        
        if is_stereo:
            for ch in range(2):
                for i in range(len(samples)):
                    processed_samples[i, ch] = samples[i, ch] * (0.3 + 0.7 * bass_gain[i])
        else:
            for i in range(len(samples)):
                processed_samples[i] = samples[i] * (0.3 + 0.7 * bass_gain[i])
        
        if is_stereo:
            processed_samples = processed_samples.flatten()
        
        processed_samples = np.clip(processed_samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(
            processed_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
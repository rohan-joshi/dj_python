#!/usr/bin/env python3

import numpy as np
import librosa
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter
from scipy import signal
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedTransitions:
    """
    Advanced transition system focusing on smooth mixing, perfect cue points, and intelligent looping
    """
    
    def __init__(self):
        self.transition_cache = {}
        self.beat_cache = {}
        self.phrase_cache = {}
        
    def find_perfect_cue_points(self, track_analysis: Dict) -> Dict[str, int]:
        """Find the perfect cue points for mixing"""
        logger.debug("Finding perfect cue points")
        
        bpm = track_analysis.get('tempo', track_analysis.get('bpm', 120))
        beats = track_analysis.get('beats', [])
        audio_features = track_analysis.get('audio_features', {})
        
        # Calculate key timing metrics
        beat_duration_ms = (60 / bpm) * 1000
        phrase_duration_ms = beat_duration_ms * 16  # 16-beat phrases
        
        cue_points = {}
        
        # 1. Intro cue point - start of first full phrase after intro
        intro_end = self._find_intro_end(track_analysis)
        first_phrase = intro_end + (phrase_duration_ms - (intro_end % phrase_duration_ms))
        cue_points['intro_cue'] = int(first_phrase)
        
        # 2. Main cue point - strongest musical phrase
        main_cue = self._find_main_phrase_start(track_analysis)
        cue_points['main_cue'] = int(main_cue)
        
        # 3. Breakdown/bridge cue - energy dip section
        breakdown_cue = self._find_breakdown_section(track_analysis)
        if breakdown_cue:
            cue_points['breakdown_cue'] = int(breakdown_cue)
        
        # 4. Drop/climax cue - highest energy section
        drop_cue = self._find_drop_section(track_analysis)
        if drop_cue:
            cue_points['drop_cue'] = int(drop_cue)
        
        # 5. Outro cue - good exit point before actual ending
        outro_cue = self._find_outro_start(track_analysis)
        cue_points['outro_cue'] = int(outro_cue)
        
        # 6. Loop cue points - sections perfect for looping
        loop_points = self._find_loop_sections(track_analysis)
        cue_points['loop_sections'] = loop_points
        
        logger.debug(f"Found cue points: {list(cue_points.keys())}")
        return cue_points
    
    def _find_intro_end(self, track_analysis: Dict) -> float:
        """Find where the intro ends and main content begins"""
        duration = track_analysis.get('duration', 0)
        duration_ms = duration * 1000
        
        # Typically intro is 16-32 bars, estimate based on energy
        energy_curve = track_analysis.get('energy_curve', [])
        if len(energy_curve) == 0:
            # Default estimate: 16 bars
            bpm = track_analysis.get('tempo', track_analysis.get('bpm', 120))
            return (60 / bpm) * 1000 * 16
        
        # Find first significant energy increase
        for i in range(min(len(energy_curve), int(duration_ms // 1000))):
            if i > 10 and energy_curve[i] > np.mean(energy_curve[:i]) * 1.3:
                return i * 1000  # Convert to ms
        
        # Fallback: 16 bars
        bpm = track_analysis.get('tempo', track_analysis.get('bpm', 120))
        return (60 / bpm) * 1000 * 16
    
    def _find_main_phrase_start(self, track_analysis: Dict) -> float:
        """Find the start of the main musical phrase"""
        duration = track_analysis.get('duration', 0)
        duration_ms = duration * 1000
        bpm = track_analysis.get('tempo', track_analysis.get('bpm', 120))
        
        # Look for the strongest beat/phrase in the first half
        beats = track_analysis.get('beats', [])
        if len(beats) == 0:
            # Default: after intro, start of first main phrase
            intro_end = self._find_intro_end(track_analysis)
            phrase_duration = (60 / bpm) * 1000 * 16
            return intro_end + phrase_duration
        
        # Find beat with strongest onset
        beat_strengths = []
        for beat in beats:
            if beat < duration_ms / 2:  # Only consider first half
                # Simple strength estimation
                strength = 1.0  # Would need onset strength analysis
                beat_strengths.append((beat, strength))
        
        if beat_strengths:
            # Return the strongest beat aligned to phrase boundary
            strongest_beat = max(beat_strengths, key=lambda x: x[1])[0]
            phrase_duration = (60 / bpm) * 1000 * 16
            phrase_aligned = strongest_beat - (strongest_beat % phrase_duration)
            return phrase_aligned
        
        # Fallback
        return self._find_intro_end(track_analysis) + (60 / bpm) * 1000 * 16
    
    def _find_breakdown_section(self, track_analysis: Dict) -> Optional[float]:
        """Find breakdown/bridge section with lower energy"""
        energy_curve = track_analysis.get('energy_curve', [])
        if len(energy_curve) == 0:
            return None
        
        # Look for sustained energy dips in middle 50% of track
        start_idx = len(energy_curve) // 4
        end_idx = 3 * len(energy_curve) // 4
        
        avg_energy = np.mean(energy_curve)
        min_duration = 8  # Minimum 8 seconds for breakdown
        
        for i in range(start_idx, end_idx - min_duration):
            section_energy = np.mean(energy_curve[i:i + min_duration])
            if section_energy < avg_energy * 0.7:  # 30% below average
                return i * 1000  # Convert to ms
        
        return None
    
    def _find_drop_section(self, track_analysis: Dict) -> Optional[float]:
        """Find the main drop/climax section"""
        energy_curve = track_analysis.get('energy_curve', [])
        if len(energy_curve) == 0:
            return None
        
        # Find highest sustained energy section
        window_size = 8  # 8 second window
        max_energy = 0
        best_position = None
        
        for i in range(len(energy_curve) - window_size):
            window_energy = np.mean(energy_curve[i:i + window_size])
            if window_energy > max_energy:
                max_energy = window_energy
                best_position = i
        
        return best_position * 1000 if best_position else None
    
    def _find_outro_start(self, track_analysis: Dict) -> float:
        """Find good outro starting point"""
        duration = track_analysis.get('duration', 0)
        duration_ms = duration * 1000
        bpm = track_analysis.get('tempo', track_analysis.get('bpm', 120))
        
        # Typically outros are last 16-32 bars
        outro_duration = (60 / bpm) * 1000 * 24  # 24 bars
        outro_start = duration_ms - outro_duration
        
        # Align to phrase boundary
        phrase_duration = (60 / bpm) * 1000 * 16
        outro_aligned = outro_start - (outro_start % phrase_duration)
        
        return max(0, outro_aligned)
    
    def _find_loop_sections(self, track_analysis: Dict) -> List[Dict]:
        """Find sections perfect for looping"""
        bpm = track_analysis.get('tempo', track_analysis.get('bpm', 120))
        duration = track_analysis.get('duration', 0)
        duration_ms = duration * 1000
        
        loop_sections = []
        phrase_duration = (60 / bpm) * 1000 * 16  # 16-beat phrases
        
        # Standard loop sections
        intro_end = self._find_intro_end(track_analysis)
        main_start = self._find_main_phrase_start(track_analysis)
        outro_start = self._find_outro_start(track_analysis)
        
        # Intro loop (first 16 bars after intro)
        loop_sections.append({
            'name': 'intro_loop',
            'start_ms': intro_end,
            'end_ms': intro_end + phrase_duration,
            'type': 'intro'
        })
        
        # Main loop (main phrase)
        loop_sections.append({
            'name': 'main_loop',
            'start_ms': main_start,
            'end_ms': main_start + phrase_duration,
            'type': 'main'
        })
        
        # Pre-outro loop
        if outro_start > main_start + phrase_duration * 2:
            loop_sections.append({
                'name': 'pre_outro_loop',
                'start_ms': outro_start - phrase_duration,
                'end_ms': outro_start,
                'type': 'outro'
            })
        
        # Breakdown loop (if available)
        breakdown = self._find_breakdown_section(track_analysis)
        if breakdown:
            loop_sections.append({
                'name': 'breakdown_loop',
                'start_ms': breakdown,
                'end_ms': breakdown + phrase_duration,
                'type': 'breakdown'
            })
        
        return loop_sections
    
    def create_smooth_transition(self, track1: AudioSegment, track2: AudioSegment,
                               track1_analysis: Dict, track2_analysis: Dict,
                               transition_style: str = 'crossfade') -> Dict:
        """Create ultra-smooth transition between tracks"""
        logger.info(f"Creating smooth {transition_style} transition")
        
        # Get perfect cue points
        track1_cues = self.find_perfect_cue_points(track1_analysis)
        track2_cues = self.find_perfect_cue_points(track2_analysis)
        
        # Choose optimal transition points based on transition style
        exit_point = self._choose_exit_point(track1, track1_analysis, track1_cues)
        entry_point = self._choose_entry_point(track2, track2_analysis, track2_cues, transition_style)
        
        # Create the transition based on style
        if transition_style == 'crossfade':
            result = self._create_crossfade_transition(
                track1, track2, exit_point, entry_point, 
                track1_analysis, track2_analysis
            )
        elif transition_style == 'loop_roll':
            result = self._create_loop_roll_transition(
                track1, track2, exit_point, entry_point,
                track1_analysis, track2_analysis
            )
        elif transition_style == 'quick_cut':
            result = self._create_quick_cut_transition(
                track1, track2, exit_point, entry_point,
                track1_analysis, track2_analysis
            )
        elif transition_style == 'filter_sweep':
            result = self.create_filter_sweep_transition(
                track1, track2, track1_analysis, track2_analysis,
                exit_point, entry_point
            )
        elif transition_style == 'echo_delay':
            result = self.create_echo_delay_transition(
                track1, track2, track1_analysis, track2_analysis,
                exit_point, entry_point
            )
        elif transition_style == 'drop_aligned':
            result = self.create_drop_aligned_transition(
                track1, track2, track1_analysis, track2_analysis
            )
        else:
            # Default to crossfade
            result = self._create_crossfade_transition(
                track1, track2, exit_point, entry_point,
                track1_analysis, track2_analysis
            )
        
        return result
    
    def _choose_exit_point(self, track: AudioSegment, analysis: Dict, cues: Dict) -> int:
        """Choose optimal exit point from current track"""
        # Prefer outro cue, but consider track length
        if len(track) > 120000:  # If longer than 2 minutes
            return cues.get('outro_cue', len(track) - 32000)
        else:
            # For shorter tracks, exit around 75% mark
            return int(len(track) * 0.75)
    
    def _choose_entry_point(self, track: AudioSegment, analysis: Dict, cues: Dict, transition_style: str = 'crossfade') -> int:
        """Choose optimal entry point based on transition type"""
        if transition_style == 'loop_roll':
            return self._find_loop_roll_entry_point(track, analysis, cues)
        elif transition_style == 'drop_aligned':
            return self._find_drop_aligned_entry_point(track, analysis, cues)
        elif transition_style == 'filter_sweep':
            return self._find_filter_sweep_entry_point(track, analysis, cues)
        elif transition_style == 'echo_delay':
            return self._find_echo_entry_point(track, analysis, cues)
        else:
            # Default crossfade entry
            return cues.get('main_cue', cues.get('intro_cue', 16000))
    
    def _find_loop_roll_entry_point(self, track: AudioSegment, analysis: Dict, cues: Dict) -> int:
        """Find the most impactful entry point for loop roll transitions"""
        # Loop rolls should hit the biggest moment - prioritize drops, then main phrases
        drop_cue = cues.get('drop_cue')
        if drop_cue:
            return drop_cue
        
        # Look for strong energy increases (drops we might have missed)
        structure = self.detect_enhanced_drops_and_breakdowns(analysis)
        if structure['drops']:
            # Use the first high-confidence drop
            for drop in structure['drops']:
                if drop['confidence'] > 0.6:
                    return drop['time_ms']
        
        # Fallback: main phrase with strong energy
        main_cue = cues.get('main_cue', 16000)
        
        # Adjust to nearest high-energy moment
        energy_curve = analysis.get('energy_curve', [])
        if len(energy_curve) > 0:
            # Convert main_cue to energy curve index
            duration = analysis.get('duration', 0)
            time_per_sample = duration / len(energy_curve)
            main_idx = int(main_cue / 1000 / time_per_sample)
            
            # Look for energy peak within ±8 seconds
            search_window = int(8 / time_per_sample)
            start_idx = max(0, main_idx - search_window)
            end_idx = min(len(energy_curve), main_idx + search_window)
            
            if end_idx > start_idx:
                window = energy_curve[start_idx:end_idx]
                peak_idx = start_idx + np.argmax(window)
                peak_time_ms = int(peak_idx * time_per_sample * 1000)
                return peak_time_ms
        
        return main_cue
    
    def _find_drop_aligned_entry_point(self, track: AudioSegment, analysis: Dict, cues: Dict) -> int:
        """Find entry point that sets up perfect drop alignment"""
        structure = self.detect_enhanced_drops_and_breakdowns(analysis)
        
        # For drop alignment, we want to start early enough to build up to the drop
        if structure['drops']:
            best_drop = None
            for drop in structure['drops']:
                if drop['confidence'] > 0.7:
                    best_drop = drop
                    break
            
            if best_drop:
                # Start 16-24 bars before the drop to allow for buildup
                bpm = analysis.get('tempo', analysis.get('bpm', 120))
                buildup_duration = int((60 / bpm) * 1000 * 20)  # 20 bars
                
                entry_point = max(0, best_drop['time_ms'] - buildup_duration)
                
                # Align to phrase boundary
                beats = self.detect_precise_beats_and_phrases(track, analysis)
                phrase_boundaries = beats.get('phrase_boundaries_ms', [])
                
                if phrase_boundaries:
                    # Find nearest phrase boundary
                    closest_phrase = min(phrase_boundaries, key=lambda x: abs(x - entry_point))
                    if abs(closest_phrase - entry_point) < 4000:  # Within 4 seconds
                        return closest_phrase
                
                return entry_point
        
        # Fallback to main cue
        return cues.get('main_cue', cues.get('intro_cue', 16000))
    
    def _find_filter_sweep_entry_point(self, track: AudioSegment, analysis: Dict, cues: Dict) -> int:
        """Find harmonically and energetically appropriate entry for filter sweeps"""
        # Filter sweeps work well when starting at the beginning of a musical phrase
        # but not necessarily the very beginning of the track
        
        # Look for the first strong musical phrase after any intro
        intro_end = self._find_intro_end(analysis)
        
        # Find the first phrase boundary after intro
        beats = self.detect_precise_beats_and_phrases(track, analysis)
        phrase_boundaries = beats.get('phrase_boundaries_ms', [])
        
        for phrase_start in phrase_boundaries:
            if phrase_start > intro_end + 8000:  # At least 8 seconds after intro
                return int(phrase_start)
        
        # Fallback: main cue adjusted to phrase boundary
        main_cue = cues.get('main_cue', 16000)
        if phrase_boundaries:
            closest_phrase = min(phrase_boundaries, key=lambda x: abs(x - main_cue))
            return closest_phrase
        
        return main_cue
    
    def _find_echo_entry_point(self, track: AudioSegment, analysis: Dict, cues: Dict) -> int:
        """Find gentle entry point for echo/delay transitions"""
        # Echo transitions work well with softer entrances
        # Look for breakdown sections or lower energy moments
        
        structure = self.detect_enhanced_drops_and_breakdowns(analysis)
        
        # Prefer starting at a breakdown or lower energy section
        if structure['breakdowns']:
            return structure['breakdowns'][0]['time_ms']
        
        # Look for lower energy sections in first half of track
        energy_curve = analysis.get('energy_curve', [])
        if len(energy_curve) > 0:
            duration = analysis.get('duration', 0)
            time_per_sample = duration / len(energy_curve)
            
            # Search first 50% of track for lower energy moments
            half_point = len(energy_curve) // 2
            first_half = energy_curve[:half_point]
            
            if len(first_half) > 10:
                avg_energy = np.mean(first_half)
                
                # Find sections with below-average energy
                for i, energy in enumerate(first_half[10:], 10):  # Skip very beginning
                    if energy < avg_energy * 0.8:  # 20% below average
                        time_ms = int(i * time_per_sample * 1000)
                        if time_ms > 16000:  # At least 16 seconds in
                            return time_ms
        
        # Fallback: main cue
        return cues.get('main_cue', cues.get('intro_cue', 16000))
    
    def _create_crossfade_transition(self, track1: AudioSegment, track2: AudioSegment,
                                   exit_point: int, entry_point: int,
                                   analysis1: Dict, analysis2: Dict) -> Dict:
        """Create perfect crossfade transition"""
        
        # Calculate optimal crossfade duration based on BPMs
        bpm1 = analysis1.get('tempo', analysis1.get('bpm', 120))
        bpm2 = analysis2.get('tempo', analysis2.get('bpm', 120))
        
        # Use phrase-aligned crossfade duration
        slower_bpm = min(bpm1, bpm2)
        bars_to_cross = 8  # 8-bar crossfade for smoothness
        crossfade_duration = int((60 / slower_bpm) * 1000 * bars_to_cross)
        
        # Ensure crossfade doesn't exceed available audio
        max_crossfade = min(len(track1) - exit_point, len(track2) - entry_point)
        crossfade_duration = min(crossfade_duration, max_crossfade, 16000)  # Max 16 seconds
        
        # Extract segments
        track1_outro = track1[exit_point:exit_point + crossfade_duration]
        track2_intro = track2[entry_point:entry_point + crossfade_duration]
        
        # Apply beatmatching if needed
        if abs(bpm1 - bpm2) > 2:  # Only if significant difference
            track2_intro = self._apply_beatmatching(track2_intro, bpm2, bpm1)
        
        # Ensure channel compatibility
        track1_outro, track2_intro = self._match_channels(track1_outro, track2_intro)
        
        # Create smooth crossfade curve
        crossfade = self._create_perfect_crossfade(track1_outro, track2_intro)
        
        # Assemble final mix
        before_transition = track1[:exit_point]
        after_transition = track2[entry_point + crossfade_duration:]
        
        mixed_track = before_transition + crossfade + after_transition
        
        return {
            'audio': mixed_track,
            'transition_start': exit_point,
            'transition_end': exit_point + crossfade_duration,
            'crossfade_duration': crossfade_duration,
            'method': 'crossfade'
        }
    
    def _create_loop_roll_transition(self, track1: AudioSegment, track2: AudioSegment,
                                   exit_point: int, entry_point: int,
                                   analysis1: Dict, analysis2: Dict) -> Dict:
        """Create transition with loop roll effect"""
        logger.debug("Creating loop roll transition")
        
        # Find good loop section in track1
        cues1 = self.find_perfect_cue_points(analysis1)
        loop_sections = cues1.get('loop_sections', [])
        
        # Choose best loop section for transition
        best_loop = None
        for loop_section in loop_sections:
            if loop_section['start_ms'] < exit_point:
                best_loop = loop_section
        
        if not best_loop:
            # Fallback: create a loop from current position
            bpm = analysis1.get('tempo', analysis1.get('bpm', 120))
            loop_duration = int((60 / bpm) * 1000 * 4)  # 4-beat loop
            best_loop = {
                'start_ms': max(0, exit_point - loop_duration),
                'end_ms': exit_point,
                'type': 'custom'
            }
        
        # Extract loop
        loop_start = int(best_loop['start_ms'])
        loop_end = int(best_loop['end_ms'])
        loop_audio = track1[loop_start:loop_end]
        
        # Enhanced progressive loop roll following your guide specifications
        beats1 = self.detect_precise_beats_and_phrases(track1, analysis1)
        bpm_precise = beats1['tempo']
        
        # Calculate precise timing
        bar_duration_ms = int((60 / bpm_precise) * 1000 * 4)  # 4 beats per bar
        
        # Start loop roll 16 bars before exit point
        loop_roll_start = max(0, exit_point - (bar_duration_ms * 16))
        
        # Use 4-bar loop for progression
        loop_4bar = track1[loop_start:loop_start + min(bar_duration_ms * 4, loop_end - loop_start)]
        
        # Create progressive loop roll sequence according to guide
        loop_roll = AudioSegment.empty()
        
        # Bars 1-8: 4-bar loop (2 repetitions)
        for _ in range(2):
            loop_roll += loop_4bar
        
        # Bars 9-12: 2-bar loop (2 repetitions)
        loop_2bar = loop_4bar[:bar_duration_ms * 2]
        for _ in range(2):
            loop_roll += loop_2bar
        
        # Bars 13-14: 1-bar loop (2 repetitions)
        loop_1bar = loop_4bar[:bar_duration_ms]
        for _ in range(2):
            loop_roll += loop_1bar
        
        # Bar 15: 1/2-bar loop (2 repetitions)
        loop_half_bar = loop_4bar[:bar_duration_ms // 2]
        for _ in range(2):
            loop_roll += loop_half_bar
        
        # Bar 16: 1/4-bar loop (4 repetitions for rapid stutter)
        loop_quarter_bar = loop_4bar[:bar_duration_ms // 4]
        for _ in range(4):
            loop_roll += loop_quarter_bar
        
        # Apply automation and effects
        loop_roll = self._apply_loop_roll_automation(loop_roll, bpm_precise)
        
        # Add subtle reverb tail for smooth cut to track2
        loop_roll_with_tail = self._add_transition_tail(loop_roll, 300)  # 300ms tail
        
        # Cut to track2 on final beat
        track2_start = track2[entry_point:]
        
        # Assemble final mix
        before_loop = track1[:loop_roll_start]
        mixed_track = before_loop + loop_roll_with_tail + track2_start
        
        return {
            'audio': mixed_track,
            'transition_start': loop_roll_start,
            'transition_end': loop_roll_start + len(loop_roll_with_tail),
            'loop_duration': len(loop_roll),
            'method': 'loop_roll'
        }
    
    def _create_quick_cut_transition(self, track1: AudioSegment, track2: AudioSegment,
                                   exit_point: int, entry_point: int,
                                   analysis1: Dict, analysis2: Dict) -> Dict:
        """Create quick cut transition on beat"""
        logger.debug("Creating quick cut transition")
        
        # Ensure cut happens exactly on beat
        bpm1 = analysis1.get('tempo', analysis1.get('bpm', 120))
        beat_duration = (60 / bpm1) * 1000
        
        # Align exit point to nearest beat
        beat_aligned_exit = round(exit_point / beat_duration) * beat_duration
        beat_aligned_exit = int(beat_aligned_exit)
        
        # Simple cut
        before_cut = track1[:beat_aligned_exit]
        after_cut = track2[entry_point:]
        
        mixed_track = before_cut + after_cut
        
        return {
            'audio': mixed_track,
            'transition_start': beat_aligned_exit,
            'transition_end': beat_aligned_exit,
            'cut_point': beat_aligned_exit,
            'method': 'quick_cut'
        }
    
    def _apply_beatmatching(self, audio: AudioSegment, source_bpm: float, target_bpm: float) -> AudioSegment:
        """Apply beatmatching without obvious artifacts"""
        if abs(source_bpm - target_bpm) < 0.5:
            return audio
        
        # Use natural tempo relationships when possible
        ratio = target_bpm / source_bpm
        
        # Check for natural ratios (2:1, 3:2, 4:3, etc.)
        natural_ratios = [0.5, 2/3, 0.75, 1.0, 1.33, 1.5, 2.0]
        closest_ratio = min(natural_ratios, key=lambda x: abs(x - ratio))
        
        if abs(ratio - closest_ratio) < 0.1:  # Within 10%
            ratio = closest_ratio
        
        # Apply pitch shifting with proper channel handling
        samples = audio.get_array_of_samples()
        samples = np.array(samples).astype(np.float32)
        
        # Handle stereo vs mono properly
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        
        # Simple time stretching (in practice, use more sophisticated algorithms)
        if ratio != 1.0:
            if audio.channels == 2:
                # Process each channel separately
                new_length = int(len(samples) / ratio)
                stretched_left = np.interp(
                    np.linspace(0, len(samples) - 1, new_length),
                    np.arange(len(samples)),
                    samples[:, 0]
                )
                stretched_right = np.interp(
                    np.linspace(0, len(samples) - 1, new_length),
                    np.arange(len(samples)),
                    samples[:, 1]
                )
                samples = np.column_stack((stretched_left, stretched_right))
                samples = samples.flatten()
            else:
                # Mono processing
                new_length = int(len(samples) / ratio)
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, new_length),
                    np.arange(len(samples)),
                    samples
                )
        else:
            if audio.channels == 2:
                samples = samples.flatten()
        
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(
            samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
    
    def _create_perfect_crossfade(self, audio1: AudioSegment, audio2: AudioSegment) -> AudioSegment:
        """Create perfect equal-power crossfade"""
        # Ensure both segments are same length and channel count
        min_length = min(len(audio1), len(audio2))
        audio1 = audio1[:min_length]
        audio2 = audio2[:min_length]
        
        # Ensure same channel count
        audio1, audio2 = self._match_channels(audio1, audio2)
        
        # Convert to numpy for processing
        samples1 = np.array(audio1.get_array_of_samples()).astype(np.float32)
        samples2 = np.array(audio2.get_array_of_samples()).astype(np.float32)
        
        # Handle channel mismatches at the sample level
        if audio1.channels == 2:
            samples1 = samples1.reshape((-1, 2))
        if audio2.channels == 2:
            samples2 = samples2.reshape((-1, 2))
            
        # If one is stereo and one is mono, convert mono to stereo
        if audio1.channels == 2 and audio2.channels == 1:
            # Convert samples2 to stereo by duplicating the mono channel
            samples2 = np.column_stack((samples2, samples2))
        elif audio1.channels == 1 and audio2.channels == 2:
            # Convert samples1 to stereo by duplicating the mono channel
            samples1 = np.column_stack((samples1, samples1))
        
        # Now both should have same structure - flatten for length comparison
        if len(samples1.shape) > 1:
            samples1_flat = samples1.flatten()
        else:
            samples1_flat = samples1
            
        if len(samples2.shape) > 1:
            samples2_flat = samples2.flatten()
        else:
            samples2_flat = samples2
        
        # Ensure same array length (this is critical!)
        min_samples = min(len(samples1_flat), len(samples2_flat))
        samples1 = samples1_flat[:min_samples]
        samples2 = samples2_flat[:min_samples]
        
        # Create equal-power crossfade curve
        crossfade_length = len(samples1)
        fade_curve = np.linspace(0, 1, crossfade_length)
        
        # Equal power curves
        gain1 = np.cos(fade_curve * np.pi / 2)  # Cosine fade out
        gain2 = np.sin(fade_curve * np.pi / 2)  # Sine fade in
        
        # Apply crossfade to flattened samples
        mixed_samples = (samples1 * gain1) + (samples2 * gain2)
        
        mixed_samples = np.clip(mixed_samples, -32768, 32767).astype(np.int16)
        
        # Determine the correct channel count (use the higher of the two)
        result_channels = max(audio1.channels, audio2.channels)
        
        return AudioSegment(
            mixed_samples.tobytes(),
            frame_rate=audio1.frame_rate,
            sample_width=audio1.sample_width,
            channels=result_channels
        )
    
    def detect_precise_beats_and_phrases(self, audio: AudioSegment, track_analysis: Dict) -> Dict:
        """Detect beats and phrase boundaries with high precision"""
        audio_key = f"beats_{len(audio)}_{track_analysis.get('tempo', 120)}"
        if audio_key in self.beat_cache:
            return self.beat_cache[audio_key]
        
        # Convert to librosa format
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        samples = samples / np.max(np.abs(samples))  # Normalize
        
        sr = audio.frame_rate
        
        # Detect beats with high precision
        tempo, beat_frames = librosa.beat.beat_track(
            y=samples, sr=sr, hop_length=512, units='samples'
        )
        
        # Convert beat frames to milliseconds
        beat_times_ms = (beat_frames / sr * 1000).astype(int)
        
        # Detect downbeats (start of measures)
        try:
            downbeat_frames = librosa.beat.beat_track(
                y=samples, sr=sr, hop_length=512, units='samples', trim=False
            )[1]
            
            # Estimate 4/4 time signature downbeats
            downbeats_ms = []
            beats_per_bar = 4
            for i in range(0, len(beat_times_ms), beats_per_bar):
                if i < len(beat_times_ms):
                    downbeats_ms.append(beat_times_ms[i])
                    
        except Exception:
            # Fallback: assume every 4th beat is a downbeat
            downbeats_ms = beat_times_ms[::4].tolist()
        
        # Detect phrase boundaries (typically 16 beats = 4 bars)
        phrase_boundaries = []
        beats_per_phrase = 16
        for i in range(0, len(beat_times_ms), beats_per_phrase):
            if i < len(beat_times_ms):
                phrase_boundaries.append(beat_times_ms[i])
        
        # Add energy-based phrase detection
        energy_phrases = self._detect_energy_based_phrases(samples, sr, beat_times_ms)
        
        # Combine musical and energy-based phrase boundaries
        all_phrases = sorted(set(phrase_boundaries + energy_phrases))
        
        result = {
            'beats_ms': beat_times_ms.tolist(),
            'downbeats_ms': downbeats_ms,
            'phrase_boundaries_ms': all_phrases,
            'tempo': float(tempo),
            'beats_per_bar': 4,
            'bars_per_phrase': 4
        }
        
        self.beat_cache[audio_key] = result
        return result
    
    def _detect_energy_based_phrases(self, samples: np.ndarray, sr: int, beat_times_ms: List[int]) -> List[int]:
        """Detect phrase boundaries based on energy changes"""
        # Calculate RMS energy in 2-bar windows
        window_size_ms = 8000  # 8 seconds ≈ 2 bars at 120 BPM
        hop_size_ms = 2000     # 2 seconds
        
        phrase_boundaries = []
        
        for i in range(0, len(samples) - int(window_size_ms * sr / 1000), int(hop_size_ms * sr / 1000)):
            start_sample = i
            end_sample = i + int(window_size_ms * sr / 1000)
            
            if end_sample < len(samples):
                window = samples[start_sample:end_sample]
                window_energy = np.sqrt(np.mean(window ** 2))
                
                # Check for significant energy change (>40% change)
                if i > 0:
                    prev_start = max(0, i - int(hop_size_ms * sr / 1000))
                    prev_end = i
                    prev_window = samples[prev_start:prev_end]
                    prev_energy = np.sqrt(np.mean(prev_window ** 2))
                    
                    if prev_energy > 0:
                        energy_change = abs(window_energy - prev_energy) / prev_energy
                        if energy_change > 0.4:  # 40% change
                            time_ms = int(i / sr * 1000)
                            # Align to nearest beat
                            nearest_beat = min(beat_times_ms, key=lambda x: abs(x - time_ms))
                            if abs(nearest_beat - time_ms) < 500:  # Within 500ms
                                phrase_boundaries.append(nearest_beat)
        
        return phrase_boundaries
    
    def create_filter_sweep_transition(self, track1: AudioSegment, track2: AudioSegment,
                                     track1_analysis: Dict, track2_analysis: Dict,
                                     exit_point: int, entry_point: int) -> Dict:
        """Create progressive filter sweep transition"""
        logger.info("Creating filter sweep transition")
        
        # Get beat information
        beats1 = self.detect_precise_beats_and_phrases(track1, track1_analysis)
        beats2 = self.detect_precise_beats_and_phrases(track2, track2_analysis)
        
        # Calculate sweep duration (8-16 bars)
        bpm = beats1['tempo']
        bars_to_sweep = 12  # 12 bars for smooth sweep
        sweep_duration_ms = int((60 / bpm) * 1000 * bars_to_sweep)
        
        # Extract segments
        track1_outro = track1[exit_point:exit_point + sweep_duration_ms]
        track2_intro = track2[entry_point:entry_point + sweep_duration_ms]
        
        # Create progressive filter sweep on track1
        filtered_track1 = self._apply_progressive_filter_sweep(track1_outro, sweep_duration_ms, 'high_pass')
        
        # Create complementary filter on track2
        filtered_track2 = self._apply_progressive_filter_sweep(track2_intro, sweep_duration_ms, 'low_pass_intro')
        
        # Create crossfade
        crossfaded = self._create_perfect_crossfade(filtered_track1, filtered_track2)
        
        # Assemble final mix
        before_transition = track1[:exit_point]
        after_transition = track2[entry_point + sweep_duration_ms:]
        
        mixed_track = before_transition + crossfaded + after_transition
        
        return {
            'audio': mixed_track,
            'transition_start': exit_point,
            'transition_end': exit_point + sweep_duration_ms,
            'sweep_duration': sweep_duration_ms,
            'method': 'filter_sweep'
        }
    
    def _apply_progressive_filter_sweep(self, audio: AudioSegment, duration_ms: int, filter_type: str) -> AudioSegment:
        """Apply progressive filter sweep over duration"""
        result = AudioSegment.empty()
        chunk_duration = 250  # Process in 250ms chunks for smooth transition
        
        for i in range(0, duration_ms, chunk_duration):
            start_ms = i
            end_ms = min(i + chunk_duration, duration_ms)
            chunk = audio[start_ms:end_ms]
            
            if len(chunk) == 0:
                break
            
            # Calculate filter progression (0.0 to 1.0)
            progress = i / duration_ms
            
            if filter_type == 'high_pass':
                # Progressive high-pass: 20Hz -> 2000Hz
                cutoff_freq = 20 + (2000 - 20) * (progress ** 1.5)  # Exponential curve
                filtered_chunk = high_pass_filter(chunk, cutoff_freq)
                
                # Also apply volume reduction for smooth exit
                volume_reduction = 1.0 - (progress ** 2)  # Quadratic fade
                filtered_chunk = filtered_chunk + (20 * np.log10(max(0.01, volume_reduction)))
                
            elif filter_type == 'low_pass_intro':
                # Start with bass cut, gradually introduce full spectrum
                cutoff_freq = 200 + (18000 - 200) * progress  # 200Hz -> 18kHz
                if progress < 0.5:
                    # First half: remove bass
                    filtered_chunk = high_pass_filter(chunk, 200 - 180 * progress)
                else:
                    # Second half: full spectrum
                    filtered_chunk = chunk
                
                # Volume introduction
                volume_increase = progress ** 0.5  # Square root curve for smooth intro
                filtered_chunk = filtered_chunk + (20 * np.log10(max(0.01, volume_increase)))
            
            else:
                filtered_chunk = chunk
            
            result += filtered_chunk
        
        return result
    
    def create_echo_delay_transition(self, track1: AudioSegment, track2: AudioSegment,
                                   track1_analysis: Dict, track2_analysis: Dict,
                                   exit_point: int, entry_point: int) -> Dict:
        """Create echo/delay fade transition"""
        logger.info("Creating echo/delay transition")
        
        beats1 = self.detect_precise_beats_and_phrases(track1, track1_analysis)
        bpm = beats1['tempo']
        
        # Calculate delay timing
        quarter_note_ms = int((60 / bpm) * 1000)  # 1/4 note delay
        echo_build_duration = quarter_note_ms * 16  # 4 bars to build echo
        echo_tail_duration = quarter_note_ms * 8   # 2 bars of echo tail
        
        # Extract segments
        track1_outro = track1[exit_point - echo_build_duration:exit_point + echo_tail_duration]
        track2_intro = track2[entry_point:entry_point + echo_build_duration + echo_tail_duration]
        
        # Apply echo effect to track1
        echoed_track1 = self._apply_echo_effect(track1_outro, quarter_note_ms, echo_build_duration)
        
        # Create wash effect (reverb-like)
        washed_track1 = self._apply_reverb_wash(echoed_track1, echo_tail_duration)
        
        # Volume automation on track1
        automated_track1 = self._apply_echo_volume_automation(washed_track1, echo_build_duration, echo_tail_duration)
        
        # Crossfade with track2
        crossfaded = self._create_echo_crossfade(automated_track1, track2_intro, echo_build_duration)
        
        # Assemble final mix
        before_transition = track1[:exit_point - echo_build_duration]
        after_transition = track2[entry_point + echo_build_duration + echo_tail_duration:]
        
        mixed_track = before_transition + crossfaded + after_transition
        
        return {
            'audio': mixed_track,
            'transition_start': exit_point - echo_build_duration,
            'transition_end': exit_point + echo_tail_duration,
            'echo_duration': echo_build_duration + echo_tail_duration,
            'method': 'echo_delay'
        }
    
    def _apply_echo_effect(self, audio: AudioSegment, delay_ms: int, build_duration_ms: int) -> AudioSegment:
        """Apply echo effect with progressive build"""
        result = audio
        
        # Create echo delays
        delays = [delay_ms, delay_ms * 2, delay_ms * 3]  # 1/4, 1/2, 3/4 note delays
        feedbacks = [0.65, 0.45, 0.25]  # Decreasing feedback
        
        for delay, feedback in zip(delays, feedbacks):
            # Create delayed version
            silence = AudioSegment.silent(duration=delay)
            delayed = silence + audio
            
            # Apply feedback reduction
            delayed = delayed + (20 * np.log10(feedback))
            
            # Progressive echo build
            if len(delayed) > len(result):
                # Pad result to match
                result = result + AudioSegment.silent(duration=len(delayed) - len(result))
            elif len(result) > len(delayed):
                # Pad delayed to match
                delayed = delayed + AudioSegment.silent(duration=len(result) - len(delayed))
            
            # Mix with progressive intensity
            mix_intensity = min(1.0, len(result) / build_duration_ms)
            delayed = delayed + (20 * np.log10(mix_intensity))
            
            result = result.overlay(delayed)
        
        return result
    
    def _apply_reverb_wash(self, audio: AudioSegment, tail_duration_ms: int) -> AudioSegment:
        """Apply reverb wash effect"""
        # Simple reverb simulation using multiple delays
        reverb_delays = [23, 47, 83, 127, 211]  # Prime numbers for natural sound
        
        result = audio
        for delay in reverb_delays:
            silence = AudioSegment.silent(duration=delay)
            delayed = silence + audio
            
            # Reduce volume and apply damping
            delayed = delayed + (20 * np.log10(0.3))  # -10dB
            delayed = low_pass_filter(delayed, 8000)  # Damping
            
            if len(delayed) > len(result):
                result = result + AudioSegment.silent(duration=len(delayed) - len(result))
            elif len(result) > len(delayed):
                delayed = delayed + AudioSegment.silent(duration=len(result) - len(delayed))
            
            result = result.overlay(delayed)
        
        return result
    
    def _apply_echo_volume_automation(self, audio: AudioSegment, build_duration_ms: int, tail_duration_ms: int) -> AudioSegment:
        """Apply volume automation for echo transition"""
        total_duration = len(audio)
        
        # Create volume curve
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        
        # Build phase: gradually introduce echo
        build_samples = int(build_duration_ms * audio.frame_rate / 1000)
        for i in range(min(build_samples, len(samples))):
            progress = i / build_samples
            volume_multiplier = 1.0 - (progress * 0.3)  # Gradually reduce dry signal
            if audio.channels == 2:
                samples[i] *= volume_multiplier
            else:
                samples[i] *= volume_multiplier
        
        # Tail phase: fade out
        tail_start = max(0, len(samples) - int(tail_duration_ms * audio.frame_rate / 1000))
        for i in range(tail_start, len(samples)):
            progress = (i - tail_start) / (len(samples) - tail_start)
            volume_multiplier = 1.0 - progress  # Linear fade out
            if audio.channels == 2:
                samples[i] *= volume_multiplier
            else:
                samples[i] *= volume_multiplier
        
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(
            samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
    
    def _create_echo_crossfade(self, echoed_audio: AudioSegment, incoming_audio: AudioSegment, crossfade_duration_ms: int) -> AudioSegment:
        """Create crossfade for echo transition"""
        # Simple overlay for echo effect
        if len(incoming_audio) > len(echoed_audio):
            echoed_audio = echoed_audio + AudioSegment.silent(duration=len(incoming_audio) - len(echoed_audio))
        elif len(echoed_audio) > len(incoming_audio):
            incoming_audio = incoming_audio + AudioSegment.silent(duration=len(echoed_audio) - len(incoming_audio))
        
        # Progressive volume introduction for incoming track
        crossfade_samples = int(crossfade_duration_ms * incoming_audio.frame_rate / 1000)
        
        samples_incoming = np.array(incoming_audio.get_array_of_samples(), dtype=np.float32)
        if incoming_audio.channels == 2:
            samples_incoming = samples_incoming.reshape((-1, 2))
        
        # Apply volume curve to incoming track
        for i in range(min(crossfade_samples, len(samples_incoming))):
            progress = i / crossfade_samples
            volume_multiplier = progress ** 0.5  # Square root curve
            if incoming_audio.channels == 2:
                samples_incoming[i] *= volume_multiplier
            else:
                samples_incoming[i] *= volume_multiplier
        
        samples_incoming = np.clip(samples_incoming, -32768, 32767).astype(np.int16)
        
        incoming_automated = AudioSegment(
            samples_incoming.tobytes(),
            frame_rate=incoming_audio.frame_rate,
            sample_width=incoming_audio.sample_width,
            channels=incoming_audio.channels
        )
        
        return echoed_audio.overlay(incoming_automated)
    
    def _apply_loop_roll_automation(self, loop_roll: AudioSegment, bpm: float) -> AudioSegment:
        """Apply volume and filter automation to loop roll"""
        samples = np.array(loop_roll.get_array_of_samples(), dtype=np.float32)
        if loop_roll.channels == 2:
            samples = samples.reshape((-1, 2))
        
        total_samples = len(samples)
        
        # Apply progressive volume increase and high-pass filter effect
        for i in range(total_samples):
            progress = i / total_samples
            
            # Volume automation: gradually increase energy
            volume_multiplier = 0.8 + (0.4 * progress)  # 80% -> 120%
            
            # Simulate high-pass effect by reducing low frequencies in later sections
            if progress > 0.75:  # Last 25% gets aggressive filtering
                filter_intensity = (progress - 0.75) * 4  # 0 -> 1
                # Simple high-pass simulation by reducing amplitude
                high_freq_boost = 1.0 + (filter_intensity * 0.3)
                volume_multiplier *= high_freq_boost
            
            if loop_roll.channels == 2:
                samples[i] *= volume_multiplier
            else:
                samples[i] *= volume_multiplier
        
        # Add excitement with slight overdrive in final section
        final_section_start = int(total_samples * 0.85)
        for i in range(final_section_start, total_samples):
            if loop_roll.channels == 2:
                # Soft clipping for excitement
                samples[i] = np.tanh(samples[i] / 20000.0) * 25000.0
            else:
                samples[i] = np.tanh(samples[i] / 20000.0) * 25000.0
        
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(
            samples.tobytes(),
            frame_rate=loop_roll.frame_rate,
            sample_width=loop_roll.sample_width,
            channels=loop_roll.channels
        )
    
    def _add_transition_tail(self, audio: AudioSegment, tail_duration_ms: int) -> AudioSegment:
        """Add subtle reverb tail for smooth transition"""
        # Create a short reverb tail
        silence = AudioSegment.silent(duration=tail_duration_ms)
        
        # Take last 100ms and create a fade tail
        last_segment = audio[-100:]
        tail = last_segment.fade_out(tail_duration_ms // 2)
        tail = tail + (20 * np.log10(0.3))  # -10dB reverb level
        
        return audio + tail
    
    def detect_enhanced_drops_and_breakdowns(self, track_analysis: Dict) -> Dict:
        """Enhanced drop and breakdown detection for professional mixing"""
        energy_curve = track_analysis.get('energy_curve', [])
        if len(energy_curve) == 0:
            return {'drops': [], 'breakdowns': [], 'buildups': []}
        
        duration = track_analysis.get('duration', 0)
        bpm = track_analysis.get('tempo', track_analysis.get('bpm', 120))
        
        # Convert energy curve indices to time
        time_per_sample = duration / len(energy_curve)
        
        drops = []
        breakdowns = []
        buildups = []
        
        # Calculate moving averages for trend detection
        window_size = max(5, len(energy_curve) // 50)  # Adaptive window
        smoothed_energy = self._moving_average(energy_curve, window_size)
        
        # Detect sudden energy increases (drops)
        for i in range(window_size, len(smoothed_energy) - window_size):
            # Look for 50%+ energy increase over 2-8 seconds
            prev_window = smoothed_energy[max(0, i-window_size):i]
            current_window = smoothed_energy[i:i+window_size]
            
            if len(prev_window) > 0 and len(current_window) > 0:
                prev_avg = np.mean(prev_window)
                current_avg = np.mean(current_window)
                
                if prev_avg > 0 and current_avg / prev_avg > 1.5:  # 50% increase
                    time_s = i * time_per_sample
                    # Ensure it's at least 8 seconds apart from previous drops
                    if not drops or time_s - drops[-1]['time_s'] > 8:
                        drops.append({
                            'time_s': time_s,
                            'time_ms': int(time_s * 1000),
                            'energy_ratio': current_avg / prev_avg,
                            'confidence': min(1.0, (current_avg / prev_avg - 1.0) / 0.5)
                        })
        
        # Detect energy dips (breakdowns)
        for i in range(window_size, len(smoothed_energy) - window_size):
            prev_window = smoothed_energy[max(0, i-window_size):i]
            current_window = smoothed_energy[i:i+window_size*2]  # Longer window for breakdowns
            
            if len(prev_window) > 0 and len(current_window) > 0:
                prev_avg = np.mean(prev_window)
                current_avg = np.mean(current_window)
                
                if current_avg < prev_avg * 0.6:  # 40% decrease
                    time_s = i * time_per_sample
                    if not breakdowns or time_s - breakdowns[-1]['time_s'] > 16:  # 16s apart
                        breakdowns.append({
                            'time_s': time_s,
                            'time_ms': int(time_s * 1000),
                            'energy_ratio': current_avg / prev_avg,
                            'duration_estimate': window_size * 2 * time_per_sample
                        })
        
        # Detect buildups (gradual energy increases before drops)
        for drop in drops:
            # Look 16-32 bars before the drop
            bars_before = 24  # 24 bars = 96 beats
            buildup_duration_s = (60 / bpm) * bars_before
            
            drop_time_s = drop['time_s']
            buildup_start_s = max(0, drop_time_s - buildup_duration_s)
            
            start_idx = int(buildup_start_s / time_per_sample)
            end_idx = int(drop_time_s / time_per_sample)
            
            if start_idx < end_idx and end_idx < len(smoothed_energy):
                buildup_energy = smoothed_energy[start_idx:end_idx]
                
                # Check for gradual increase
                if len(buildup_energy) > 10:
                    trend = np.polyfit(range(len(buildup_energy)), buildup_energy, 1)[0]
                    if trend > 0:  # Positive trend
                        buildups.append({
                            'start_time_s': buildup_start_s,
                            'start_time_ms': int(buildup_start_s * 1000),
                            'end_time_s': drop_time_s,
                            'end_time_ms': int(drop_time_s * 1000),
                            'trend_strength': trend,
                            'leads_to_drop': True
                        })
        
        return {
            'drops': drops,
            'breakdowns': breakdowns,
            'buildups': buildups
        }
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Calculate moving average for smoothing"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        
        return smoothed
    
    def create_drop_aligned_transition(self, track1: AudioSegment, track2: AudioSegment,
                                     track1_analysis: Dict, track2_analysis: Dict) -> Dict:
        """Create transition aligned to drops for maximum impact"""
        logger.info("Creating drop-aligned transition")
        
        # Use the improved entry point detection
        track1_cues = self.find_perfect_cue_points(track1_analysis)
        track2_cues = self.find_perfect_cue_points(track2_analysis)
        
        # Get optimal entry point for drop alignment
        entry_point = self._find_drop_aligned_entry_point(track2, track2_analysis, track2_cues)
        
        # Find the best exit point from track1
        track1_structure = self.detect_enhanced_drops_and_breakdowns(track1_analysis)
        
        # Look for breakdown in track1 (good exit point)
        track1_breakdown = None
        if track1_structure['breakdowns']:
            # Choose breakdown in last 50% of track for good flow
            track1_duration_s = track1_analysis.get('duration', 0)
            for breakdown in track1_structure['breakdowns']:
                if breakdown['time_s'] > track1_duration_s * 0.4:  # Last 60% of track
                    track1_breakdown = breakdown
                    break
        
        if track1_breakdown:
            exit_point = track1_breakdown['time_ms']
        else:
            # No breakdown found, use 75% point but align to phrase
            exit_point = int(len(track1) * 0.75)
            
            # Try to align to phrase boundary
            beats1 = self.detect_precise_beats_and_phrases(track1, track1_analysis)
            phrase_boundaries = beats1.get('phrase_boundaries_ms', [])
            if phrase_boundaries:
                closest_phrase = min(phrase_boundaries, key=lambda x: abs(x - exit_point))
                if abs(closest_phrase - exit_point) < 8000:  # Within 8 seconds
                    exit_point = closest_phrase
        
        # Calculate timing for perfect energy flow
        track2_structure = self.detect_enhanced_drops_and_breakdowns(track2_analysis)
        track2_drop = None
        
        if track2_structure['drops']:
            # Find the drop we're targeting
            for drop in track2_structure['drops']:
                if drop['time_ms'] >= entry_point and drop['confidence'] > 0.6:
                    track2_drop = drop
                    break
        
        # Create energy-managed transition if we have both breakdown and drop
        if track1_breakdown and track2_drop:
            # Calculate perfect timing: breakdown should end as buildup to drop begins
            bpm2 = track2_analysis.get('tempo', track2_analysis.get('bpm', 120))
            buildup_duration = int((60 / bpm2) * 1000 * 16)  # 16 bars of buildup
            
            # Adjust timing so the energy flow is perfect
            breakdown_duration = 8000  # 8 second breakdown
            total_transition_time = breakdown_duration + buildup_duration
            
            # The drop should hit right after our transition
            target_drop_time = entry_point + (track2_drop['time_ms'] - entry_point)
            
            result = self._create_energy_managed_transition(
                track1, track2, exit_point, entry_point,
                track1_analysis, track2_analysis, track1_breakdown, track2_drop
            )
        else:
            # Fallback to crossfade with the improved entry point
            result = self._create_crossfade_transition(
                track1, track2, exit_point, entry_point,
                track1_analysis, track2_analysis
            )
        
        result['method'] = 'drop_aligned'
        return result
    
    def _create_energy_managed_transition(self, track1: AudioSegment, track2: AudioSegment,
                                        exit_point: int, entry_point: int,
                                        analysis1: Dict, analysis2: Dict,
                                        breakdown: Dict, drop: Dict) -> Dict:
        """Create transition with perfect energy management"""
        
        # Phase 1: Play breakdown section of track1
        breakdown_duration = 8000  # 8 seconds
        track1_breakdown = track1[exit_point:exit_point + breakdown_duration]
        
        # Apply breakdown processing (filter, reduce energy)
        processed_breakdown = self._process_breakdown_section(track1_breakdown)
        
        # Phase 2: Introduce track2 during buildup
        buildup_start = entry_point
        buildup_duration = drop['time_ms'] - entry_point
        track2_buildup = track2[buildup_start:buildup_start + buildup_duration]
        
        # Process buildup (gradual filter opening, volume increase)
        processed_buildup = self._process_buildup_section(track2_buildup, buildup_duration)
        
        # Phase 3: Cut to drop
        track2_drop_section = track2[drop['time_ms']:]
        
        # Create smooth energy transition
        energy_transition = self._blend_breakdown_to_buildup(processed_breakdown, processed_buildup)
        
        # Assemble final mix
        before_transition = track1[:exit_point]
        mixed_track = before_transition + energy_transition + track2_drop_section
        
        return {
            'audio': mixed_track,
            'transition_start': exit_point,
            'transition_end': exit_point + len(energy_transition),
            'breakdown_info': breakdown,
            'drop_info': drop,
            'method': 'energy_managed'
        }
    
    def _process_breakdown_section(self, audio: AudioSegment) -> AudioSegment:
        """Process breakdown section for smooth energy reduction"""
        # Apply gentle low-pass filter and volume reduction
        filtered = low_pass_filter(audio, 4000)  # Remove high frequencies
        reduced = filtered + (20 * np.log10(0.7))  # Reduce to 70% volume
        return reduced
    
    def _process_buildup_section(self, audio: AudioSegment, buildup_duration_ms: int) -> AudioSegment:
        """Process buildup section with progressive energy increase"""
        result = AudioSegment.empty()
        chunk_size = 500  # 500ms chunks
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) == 0:
                break
            
            # Calculate buildup progress
            progress = i / len(audio)
            
            # Progressive filter opening
            cutoff = 200 + (18000 - 200) * progress  # 200Hz -> 18kHz
            if cutoff < 18000:
                chunk = high_pass_filter(chunk, max(20, 200 - 180 * progress))
            
            # Progressive volume increase
            volume_mult = 0.3 + (0.7 * progress)  # 30% -> 100%
            chunk = chunk + (20 * np.log10(volume_mult))
            
            result += chunk
        
        return result
    
    def _blend_breakdown_to_buildup(self, breakdown: AudioSegment, buildup: AudioSegment) -> AudioSegment:
        """Blend breakdown and buildup sections smoothly"""
        # Overlap by 2 seconds
        overlap_duration = 2000
        
        if len(breakdown) < overlap_duration or len(buildup) < overlap_duration:
            # Simple concatenation if too short
            return breakdown + buildup
        
        # Extract overlap sections
        breakdown_outro = breakdown[-overlap_duration:]
        buildup_intro = buildup[:overlap_duration]
        
        # Create crossfade
        crossfaded = self._create_perfect_crossfade(breakdown_outro, buildup_intro)
        
        # Assemble
        breakdown_main = breakdown[:-overlap_duration]
        buildup_main = buildup[overlap_duration:]
        
        return breakdown_main + crossfaded + buildup_main
    
    def select_optimal_transition_style(self, track1_analysis: Dict, track2_analysis: Dict,
                                      transition_index: int, total_tracks: int) -> str:
        """Intelligently select the best transition style based on track characteristics"""
        
        bpm1 = track1_analysis.get('tempo', track1_analysis.get('bpm', 120))
        bpm2 = track2_analysis.get('tempo', track2_analysis.get('bpm', 120))
        key1 = track1_analysis.get('key', 'C')
        key2 = track2_analysis.get('key', 'C')
        
        # Get energy and structure information
        track1_structure = self.detect_enhanced_drops_and_breakdowns(track1_analysis)
        track2_structure = self.detect_enhanced_drops_and_breakdowns(track2_analysis)
        
        # Decision logic based on your guide
        tempo_diff = abs(bpm1 - bpm2)
        
        # Rule 1: Same genre, similar BPM (±5) -> Extended crossfade
        if tempo_diff <= 5:
            if self._tracks_harmonically_compatible(key1, key2):
                return 'crossfade'  # Extended crossfade for compatible tracks
            else:
                return 'filter_sweep'  # Filter sweep for incompatible keys
        
        # Rule 2: Energy increase needed -> Drop mix
        if track2_structure['drops'] and len(track2_structure['drops']) > 0:
            # Check if track1 has breakdowns for perfect energy flow
            if track1_structure['breakdowns']:
                return 'drop_aligned'
        
        # Rule 3: Tempo change >10 BPM -> Loop roll or echo fade
        if tempo_diff > 10:
            # Use loop roll for dramatic tempo changes
            return 'loop_roll'
        
        # Rule 4: Mid-mix transitions -> Filter sweep
        if 0.2 < transition_index / total_tracks < 0.8:
            return 'filter_sweep'
        
        # Rule 5: Echo fade for ambient/breakdown sections
        if track1_structure['breakdowns'] and not track2_structure['drops']:
            return 'echo_delay'
        
        # Default: crossfade
        return 'crossfade'
    
    def _tracks_harmonically_compatible(self, key1: str, key2: str) -> bool:
        """Check if tracks are harmonically compatible using Camelot wheel"""
        # Simplified harmonic compatibility check
        if key1 == key2:
            return True
        
        # Major/minor relationships
        major_minor_pairs = [
            ('C', 'Am'), ('G', 'Em'), ('D', 'Bm'), ('A', 'F#m'),
            ('E', 'C#m'), ('B', 'G#m'), ('F#', 'D#m'), ('Db', 'Bbm'),
            ('Ab', 'Fm'), ('Eb', 'Cm'), ('Bb', 'Gm'), ('F', 'Dm')
        ]
        
        for major, minor in major_minor_pairs:
            if (key1 == major and key2 == minor) or (key1 == minor and key2 == major):
                return True
        
        # Perfect fifth relationships (basic check)
        fifth_relationships = [
            ('C', 'G'), ('G', 'D'), ('D', 'A'), ('A', 'E'),
            ('E', 'B'), ('B', 'F#'), ('F#', 'Db'), ('Db', 'Ab'),
            ('Ab', 'Eb'), ('Eb', 'Bb'), ('Bb', 'F'), ('F', 'C')
        ]
        
        for key_a, key_b in fifth_relationships:
            if (key1 == key_a and key2 == key_b) or (key1 == key_b and key2 == key_a):
                return True
        
        return False
    
    def _match_channels(self, audio1: AudioSegment, audio2: AudioSegment) -> Tuple[AudioSegment, AudioSegment]:
        """Ensure both audio segments have the same number of channels"""
        logger.debug(f"Channel matching: audio1={audio1.channels}ch, audio2={audio2.channels}ch")
        
        if audio1.channels == audio2.channels:
            return audio1, audio2
        
        # Convert mono to stereo if needed
        if audio1.channels == 1 and audio2.channels == 2:
            logger.debug("Converting audio1 from mono to stereo")
            audio1 = audio1.set_channels(2)
        elif audio1.channels == 2 and audio2.channels == 1:
            logger.debug("Converting audio2 from mono to stereo")
            audio2 = audio2.set_channels(2)
        
        logger.debug(f"After matching: audio1={audio1.channels}ch, audio2={audio2.channels}ch")
        return audio1, audio2
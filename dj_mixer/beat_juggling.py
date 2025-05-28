#!/usr/bin/env python3

import numpy as np
import random
from pydub import AudioSegment
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BeatJuggling:
    """
    Advanced beat juggling and hot cue system for creative mixing
    """
    
    def __init__(self):
        self.hot_cues = {}
        self.beat_cache = {}
        
    def analyze_beat_structure(self, audio: AudioSegment, bpm: float) -> Dict:
        """Analyze beat structure for juggling opportunities"""
        logger.debug("Analyzing beat structure for juggling")
        
        beat_duration_ms = (60 / bpm) * 1000
        bar_duration_ms = beat_duration_ms * 4
        phrase_duration_ms = beat_duration_ms * 16
        
        duration_ms = len(audio)
        
        # Identify key structural elements
        structure = {
            'beat_duration_ms': beat_duration_ms,
            'bar_duration_ms': bar_duration_ms,
            'phrase_duration_ms': phrase_duration_ms,
            'total_beats': int(duration_ms / beat_duration_ms),
            'total_bars': int(duration_ms / bar_duration_ms),
            'total_phrases': int(duration_ms / phrase_duration_ms),
            'beat_grid': self._create_beat_grid(duration_ms, beat_duration_ms),
            'juggle_points': self._find_juggle_points(audio, beat_duration_ms, bar_duration_ms)
        }
        
        return structure
    
    def _create_beat_grid(self, duration_ms: float, beat_duration_ms: float) -> List[float]:
        """Create precise beat grid for the track"""
        beat_grid = []
        current_time = 0
        
        while current_time < duration_ms:
            beat_grid.append(current_time)
            current_time += beat_duration_ms
        
        return beat_grid
    
    def _find_juggle_points(self, audio: AudioSegment, beat_duration_ms: float, 
                          bar_duration_ms: float) -> List[Dict]:
        """Find optimal points for beat juggling"""
        juggle_points = []
        duration_ms = len(audio)
        
        # Look for strong beat/bar starts throughout the track
        current_time = 0
        bar_count = 0
        
        while current_time < duration_ms - bar_duration_ms:
            # Every 4th and 8th bar are good juggle points
            if bar_count % 4 == 0:
                juggle_points.append({
                    'time_ms': current_time,
                    'type': 'bar_start',
                    'strength': 'strong' if bar_count % 8 == 0 else 'medium',
                    'duration_options': [
                        int(beat_duration_ms),      # 1 beat
                        int(beat_duration_ms * 2),  # 2 beats
                        int(beat_duration_ms * 4),  # 1 bar
                        int(beat_duration_ms * 8)   # 2 bars
                    ]
                })
            
            current_time += bar_duration_ms
            bar_count += 1
        
        return juggle_points
    
    def create_hot_cues(self, audio: AudioSegment, track_analysis: Dict) -> Dict[str, int]:
        """Create hot cue points for instant access"""
        logger.debug("Creating hot cue points")
        
        bpm = track_analysis['bpm']
        duration_ms = len(audio)
        beat_duration_ms = (60 / bpm) * 1000
        
        hot_cues = {}
        
        # Hot Cue 1: Track start (intro)
        hot_cues['intro'] = 0
        
        # Hot Cue 2: First drop/main section (usually after 32 bars)
        first_drop = min(beat_duration_ms * 32, duration_ms * 0.25)
        hot_cues['drop1'] = int(first_drop)
        
        # Hot Cue 3: Breakdown/bridge section
        breakdown = duration_ms * 0.5  # Rough middle
        hot_cues['breakdown'] = int(breakdown)
        
        # Hot Cue 4: Second drop/climax
        second_drop = duration_ms * 0.7
        hot_cues['drop2'] = int(second_drop)
        
        # Hot Cue 5: Outro start
        outro = max(duration_ms - beat_duration_ms * 32, duration_ms * 0.8)
        hot_cues['outro'] = int(outro)
        
        # Hot Cue 6-8: Dynamic juggle points
        juggle_points = self._find_juggle_points(audio, beat_duration_ms, beat_duration_ms * 4)
        strong_points = [jp for jp in juggle_points if jp['strength'] == 'strong']
        
        if len(strong_points) >= 3:
            hot_cues['juggle1'] = int(strong_points[1]['time_ms'])  # Skip first (intro)
            hot_cues['juggle2'] = int(strong_points[2]['time_ms'])
            if len(strong_points) > 3:
                hot_cues['juggle3'] = int(strong_points[3]['time_ms'])
        
        return hot_cues
    
    def execute_beat_juggle(self, audio: AudioSegment, juggle_pattern: str,
                          hot_cues: Dict[str, int], bpm: float) -> AudioSegment:
        """Execute a beat juggling pattern"""
        logger.info(f"Executing beat juggle pattern: {juggle_pattern}")
        
        beat_duration_ms = (60 / bpm) * 1000
        
        if juggle_pattern == 'simple_repeat':
            return self._simple_repeat_juggle(audio, hot_cues, beat_duration_ms)
        elif juggle_pattern == 'two_bar_loop':
            return self._two_bar_loop_juggle(audio, hot_cues, beat_duration_ms)
        elif juggle_pattern == 'stutter_effect':
            return self._stutter_effect_juggle(audio, hot_cues, beat_duration_ms)
        elif juggle_pattern == 'break_rebuild':
            return self._break_rebuild_juggle(audio, hot_cues, beat_duration_ms)
        else:
            logger.warning(f"Unknown juggle pattern: {juggle_pattern}")
            return audio
    
    def _simple_repeat_juggle(self, audio: AudioSegment, hot_cues: Dict[str, int],
                            beat_duration_ms: float) -> AudioSegment:
        """Simple 4-beat repeat juggle"""
        if 'juggle1' not in hot_cues:
            return audio
        
        juggle_start = hot_cues['juggle1']
        repeat_length = int(beat_duration_ms * 4)  # 4 beats
        
        # Extract the section to repeat
        repeat_section = audio[juggle_start:juggle_start + repeat_length]
        
        # Create the juggled version
        juggled_audio = audio[:juggle_start]
        
        # Repeat the 4-beat section 4 times
        for _ in range(4):
            juggled_audio += repeat_section
        
        # Continue with rest of track
        juggled_audio += audio[juggle_start + repeat_length:]
        
        return juggled_audio
    
    def _two_bar_loop_juggle(self, audio: AudioSegment, hot_cues: Dict[str, int],
                           beat_duration_ms: float) -> AudioSegment:
        """Two-bar loop with variations"""
        if 'juggle1' not in hot_cues:
            return audio
        
        juggle_start = hot_cues['juggle1']
        bar_length = int(beat_duration_ms * 4)
        two_bar_length = bar_length * 2
        
        # Extract two bars
        loop_section = audio[juggle_start:juggle_start + two_bar_length]
        first_bar = loop_section[:bar_length]
        second_bar = loop_section[bar_length:]
        
        # Create juggled version
        juggled_audio = audio[:juggle_start]
        
        # Pattern: Bar1, Bar1, Bar2, Bar1, Bar2, Bar2
        pattern = [first_bar, first_bar, second_bar, first_bar, second_bar, second_bar]
        for bar in pattern:
            juggled_audio += bar
        
        # Continue with rest of track
        juggled_audio += audio[juggle_start + two_bar_length:]
        
        return juggled_audio
    
    def _stutter_effect_juggle(self, audio: AudioSegment, hot_cues: Dict[str, int],
                             beat_duration_ms: float) -> AudioSegment:
        """Create stutter effect by repeating small sections"""
        if 'juggle2' not in hot_cues:
            return audio
        
        juggle_start = hot_cues['juggle2']
        
        # Create stutters of different lengths
        stutter_lengths = [
            int(beat_duration_ms / 4),   # 16th note
            int(beat_duration_ms / 2),   # 8th note
            int(beat_duration_ms),       # Quarter note
        ]
        
        juggled_audio = audio[:juggle_start]
        current_pos = juggle_start
        
        # Create 8 beats of stutter patterns
        for beat in range(8):
            stutter_length = random.choice(stutter_lengths)
            stutter_repeats = max(1, int(beat_duration_ms / stutter_length))
            
            # Extract stutter section
            stutter_section = audio[current_pos:current_pos + stutter_length]
            
            # Repeat the stutter
            for _ in range(stutter_repeats):
                juggled_audio += stutter_section
            
            current_pos += beat_duration_ms
        
        # Continue with rest of track
        juggled_audio += audio[current_pos:]
        
        return juggled_audio
    
    def _break_rebuild_juggle(self, audio: AudioSegment, hot_cues: Dict[str, int],
                            beat_duration_ms: float) -> AudioSegment:
        """Break down and rebuild pattern"""
        if 'breakdown' not in hot_cues:
            return audio
        
        breakdown_start = hot_cues['breakdown']
        beat_length = int(beat_duration_ms)
        
        # Extract 16 beats for breaking down
        section_length = beat_length * 16
        breakdown_section = audio[breakdown_start:breakdown_start + section_length]
        
        juggled_audio = audio[:breakdown_start]
        
        # Break down pattern: 8 beats, 4 beats, 2 beats, 1 beat, then rebuild
        patterns = [
            8, 8,        # Full 8-beat phrases
            4, 4, 4, 4,  # Break into 4-beat segments
            2, 2, 2, 2,  # Break into 2-beat segments
            1, 1, 1, 1,  # Break into single beats
            2, 2, 2, 2,  # Rebuild: 2 beats
            4, 4, 4, 4,  # Rebuild: 4 beats
            8, 8         # Rebuild: 8 beats
        ]
        
        source_pos = 0
        for pattern_length in patterns:
            pattern_duration = beat_length * pattern_length
            pattern_section = breakdown_section[source_pos:source_pos + pattern_duration]
            juggled_audio += pattern_section
            
            # Advance source position cyclically
            source_pos = (source_pos + pattern_duration) % len(breakdown_section)
        
        # Continue with rest of track
        juggled_audio += audio[breakdown_start + section_length:]
        
        return juggled_audio
    
    def create_quick_cuts(self, audio: AudioSegment, cut_points: List[int],
                         bpm: float) -> AudioSegment:
        """Create quick cuts on beat for dramatic effect"""
        logger.debug(f"Creating quick cuts at {len(cut_points)} points")
        
        beat_duration_ms = (60 / bpm) * 1000
        
        # Sort cut points
        cut_points = sorted(cut_points)
        
        result_audio = AudioSegment.empty()
        last_pos = 0
        
        for cut_point in cut_points:
            # Align cut to nearest beat
            aligned_cut = round(cut_point / beat_duration_ms) * beat_duration_ms
            aligned_cut = int(aligned_cut)
            
            if aligned_cut > last_pos and aligned_cut < len(audio):
                # Add audio up to cut point
                result_audio += audio[last_pos:aligned_cut]
                
                # Add brief silence for dramatic effect
                silence_duration = random.randint(50, 200)  # 50-200ms silence
                result_audio += AudioSegment.silent(duration=silence_duration)
                
                last_pos = aligned_cut
        
        # Add remaining audio
        if last_pos < len(audio):
            result_audio += audio[last_pos:]
        
        return result_audio
    
    def create_loop_roll(self, audio: AudioSegment, loop_start_ms: int,
                        bpm: float, style: str = 'progressive') -> AudioSegment:
        """Create loop roll effect"""
        logger.debug(f"Creating {style} loop roll at {loop_start_ms}ms")
        
        beat_duration_ms = (60 / bpm) * 1000
        
        if style == 'progressive':
            return self._progressive_loop_roll(audio, loop_start_ms, beat_duration_ms)
        elif style == 'reverse':
            return self._reverse_loop_roll(audio, loop_start_ms, beat_duration_ms)
        elif style == 'stutter':
            return self._stutter_loop_roll(audio, loop_start_ms, beat_duration_ms)
        else:
            return self._progressive_loop_roll(audio, loop_start_ms, beat_duration_ms)
    
    def _progressive_loop_roll(self, audio: AudioSegment, start_ms: int,
                             beat_duration_ms: float) -> AudioSegment:
        """Progressive loop roll (getting shorter)"""
        # Start with 4-beat loop, progressively get shorter
        loop_lengths = [4, 4, 2, 2, 1, 1, 0.5, 0.5, 0.25, 0.25]
        
        result_audio = audio[:start_ms]
        current_pos = start_ms
        
        for loop_beats in loop_lengths:
            loop_duration = int(beat_duration_ms * loop_beats)
            loop_section = audio[current_pos:current_pos + loop_duration]
            result_audio += loop_section
        
        # Continue with original audio
        result_audio += audio[start_ms:]
        
        return result_audio
    
    def _reverse_loop_roll(self, audio: AudioSegment, start_ms: int,
                          beat_duration_ms: float) -> AudioSegment:
        """Reverse loop roll effect"""
        loop_duration = int(beat_duration_ms * 2)  # 2-beat loop
        loop_section = audio[start_ms:start_ms + loop_duration]
        
        result_audio = audio[:start_ms]
        
        # Create reverse roll pattern
        for i in range(8):  # 8 iterations
            if i % 2 == 0:
                result_audio += loop_section
            else:
                result_audio += loop_section.reverse()
        
        # Continue with original audio
        result_audio += audio[start_ms:]
        
        return result_audio
    
    def _stutter_loop_roll(self, audio: AudioSegment, start_ms: int,
                          beat_duration_ms: float) -> AudioSegment:
        """Stutter-style loop roll"""
        stutter_length = int(beat_duration_ms / 8)  # 32nd note stutters
        
        result_audio = audio[:start_ms]
        
        # Create 4 beats of stutters
        for beat in range(4):
            beat_start = start_ms + (beat * int(beat_duration_ms))
            stutter_section = audio[beat_start:beat_start + stutter_length]
            
            # Repeat stutter 8 times per beat
            for _ in range(8):
                result_audio += stutter_section
        
        # Continue with original audio
        result_audio += audio[start_ms + int(beat_duration_ms * 4):]
        
        return result_audio
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pydub import AudioSegment
from .audio_analyzer import AudioAnalyzer
from .harmonic_analyzer import HarmonicAnalyzer
from .advanced_transitions import AdvancedTransitions
from .professional_eq import ProfessionalEQ
from .beat_juggling import BeatJuggling
from .enhanced_beatmatching import EnhancedBeatmatching
from .professional_production import ProfessionalProduction
from .dj_elements import DJElements

logger = logging.getLogger(__name__)

class CohesiveMixBuilder:
    """
    Creates beautiful, cohesive DJ mixes that sound like professional studio productions.
    Focuses on:
    - Seamless, musical transitions
    - Intelligent track flow and energy journey  
    - Natural-sounding effects that enhance the experience
    - Proper sonic cohesion throughout the mix
    """
    
    def __init__(self):
        self.analyzer = AudioAnalyzer()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.transitions = AdvancedTransitions()
        self.eq = ProfessionalEQ()
        self.beat_juggling = BeatJuggling()
        self.beatmatching = EnhancedBeatmatching()
        self.production = ProfessionalProduction()
        self.dj_elements = DJElements()
    
    def create_beautiful_mix(self, track_analyses: Dict[str, Dict], 
                           mix_style: str = 'journey') -> AudioSegment:
        """
        Create a cohesive, beautiful-sounding mix
        
        mix_style options:
        - 'journey': Build energy arc with peaks and valleys
        - 'party': Maintain high energy throughout
        - 'chill': Smooth, relaxed progression
        - 'emotional': Focus on harmonic/key progression
        """
        
        # 1. Intelligently order tracks for best flow
        ordered_tracks = self._create_optimal_track_flow(track_analyses, mix_style)
        
        # 2. Create the mix with enhanced transitions
        mixed_audio = self._build_cohesive_mix(ordered_tracks, track_analyses, mix_style)
        
        # 3. Apply final mastering touches
        final_mix = self._apply_mix_mastering(mixed_audio)
        
        logger.info(f"Created beautiful {mix_style} mix: {len(final_mix)/1000/60:.1f} minutes")
        return final_mix
    
    def _create_optimal_track_flow(self, track_analyses: Dict[str, Dict], 
                                 mix_style: str) -> List[str]:
        """Create intelligent track ordering using enhanced beatmatching"""
        
        tracks = list(track_analyses.keys())
        track_analysis_list = list(track_analyses.values())
        
        if len(tracks) <= 2:
            return tracks
        
        # Helper function for handling energy arrays
        def get_avg_energy(analysis):
            energy = analysis.get('energy', 0.5)
            return np.mean(energy) if isinstance(energy, np.ndarray) else energy
        
        # Use enhanced beatmatching for harmonic journey
        if mix_style == 'journey':
            order_indices = self.beatmatching.create_harmonic_journey(track_analysis_list, 'energy_climb')
        elif mix_style == 'party':
            # Optimize for compatibility but keep high energy tracks together
            order_indices = self.beatmatching.create_harmonic_journey(track_analysis_list, 'energy_climb')
            # Reverse to start with high energy
            high_energy_tracks = [i for i, analysis in enumerate(track_analysis_list) 
                                if get_avg_energy(analysis) > 0.7]
            if len(high_energy_tracks) > len(track_analysis_list) // 2:
                order_indices = list(reversed(order_indices))
        elif mix_style == 'chill':
            # Start with lower energy tracks
            order_indices = self.beatmatching.create_harmonic_journey(track_analysis_list, 'energy_climb')
            # Filter for chill tracks and reorder
            chill_indices = [i for i, analysis in enumerate(track_analysis_list) 
                           if get_avg_energy(analysis) < 0.6]
            if chill_indices:
                # Prioritize chill tracks at beginning
                chill_order = [i for i in order_indices if i in chill_indices]
                other_order = [i for i in order_indices if i not in chill_indices]
                order_indices = chill_order + other_order
        elif mix_style == 'emotional':
            order_indices = self.beatmatching.create_harmonic_journey(track_analysis_list, 'harmonic_circle')
        else:
            order_indices = self.beatmatching.optimize_track_order_by_compatibility(track_analysis_list)
        
        return [tracks[i] for i in order_indices]
    
    def _build_cohesive_mix(self, ordered_tracks: List[str], 
                          track_analyses: Dict[str, Dict], 
                          mix_style: str) -> AudioSegment:
        """Build the actual mix using professional DJ techniques"""
        
        if len(ordered_tracks) == 0:
            return AudioSegment.silent(duration=1000)
        
        if len(ordered_tracks) == 1:
            audio = AudioSegment.from_file(ordered_tracks[0])
            return self._normalize_for_mix(audio)
        
        logger.info(f"Building professional {mix_style} mix with {len(ordered_tracks)} tracks")
        
        # Load all tracks
        track_audios = {}
        for track_path in ordered_tracks:
            audio = AudioSegment.from_file(track_path)
            track_audios[track_path] = self._normalize_for_mix(audio)
        
        # Create the mix using advanced transitions
        mixed_audio = None
        transition_points = []
        
        for i in range(len(ordered_tracks) - 1):
            current_track_path = ordered_tracks[i]
            next_track_path = ordered_tracks[i + 1]
            
            current_analysis = track_analyses[current_track_path]
            next_analysis = track_analyses[next_track_path]
            
            # For transitions, we need individual tracks, not the accumulated mix
            if mixed_audio is None:
                # First transition: track1 + track2
                current_audio = track_audios[current_track_path]
                next_audio = track_audios[next_track_path]
                
                # Apply beatmatching between individual tracks
                current_bpm = current_analysis.get('tempo', current_analysis.get('bpm', 120))
                next_bpm = next_analysis.get('tempo', next_analysis.get('bpm', 120))
                
                current_audio, next_audio = self.beatmatching.create_perfect_beatmatch(
                    current_audio, next_audio, current_bpm, next_bpm
                )
                
                # Create transition between the two tracks
                transition_result = self._create_professional_transition(
                    current_audio, next_audio, current_analysis, next_analysis, 
                    mix_style, i, len(ordered_tracks)
                )
                
                mixed_audio = transition_result['audio']
                
            else:
                # Subsequent transitions: append next track to existing mix
                next_audio = track_audios[next_track_path]
                
                # Get the last portion of the current mix for transition analysis
                # Use a reasonable segment (last 2 minutes) to represent the "current track"
                current_segment_duration = min(120000, len(mixed_audio))  # Max 2 minutes
                current_segment = mixed_audio[-current_segment_duration:]
                
                # Create transition between segment and next track
                transition_result = self._create_professional_transition(
                    current_segment, next_audio, current_analysis, next_analysis, 
                    mix_style, i, len(ordered_tracks)
                )
                
                # Extract the transition portion and append to mix
                transition_audio = transition_result['audio']
                transition_start = transition_result.get('transition_start', 0)
                
                # Keep most of existing mix, then add transition
                mix_keep_point = len(mixed_audio) - current_segment_duration + transition_start
                mixed_audio = mixed_audio[:mix_keep_point] + transition_audio[transition_start:]
            
            transition_points.append({
                'start_ms': transition_result.get('transition_start', 0),
                'end_ms': transition_result.get('transition_end', 0),
                'method': transition_result.get('method', 'crossfade')
            })
        
        # Add DJ elements if appropriate
        if mix_style == 'party':
            mixed_audio = self.dj_elements.add_smart_effects(
                mixed_audio, list(track_analyses.values()), mix_style
            )
        
        # Apply final production touches (disabled for now due to channel mismatch)
        # mixed_audio = self.production.add_vinyl_character(mixed_audio, 0.2)
        # mixed_audio = self.production.add_room_tone(mixed_audio, 0.1)
        
        return mixed_audio
    
    def _create_professional_transition(self, track1: AudioSegment, track2: AudioSegment,
                                      analysis1: Dict, analysis2: Dict,
                                      mix_style: str, transition_index: int, total_tracks: int) -> Dict:
        """Create professional DJ transition using all advanced techniques"""
        
        # Determine transition style based on context
        transition_style = self._choose_professional_transition_style(
            analysis1, analysis2, mix_style, transition_index, total_tracks
        )
        
        logger.debug(f"Creating {transition_style} transition ({transition_index + 1}/{total_tracks - 1})")
        
        # Analyze frequency content for EQ planning
        track1_spectrum = self.eq.analyze_frequency_spectrum(track1)
        track2_spectrum = self.eq.analyze_frequency_spectrum(track2)
        
        # Create EQ transition plan with musical timing
        eq_plan = self.eq.create_eq_transition_plan(
            track1_spectrum, track2_spectrum, 8000, analysis1, analysis2
        )
        
        # Apply EQ transition
        track1_eq, track2_eq = self.eq.apply_eq_transition(
            track1, track2, eq_plan, 8000
        )
        
        # Create the main transition using advanced techniques
        transition_result = self.transitions.create_smooth_transition(
            track1_eq, track2_eq, analysis1, analysis2, transition_style
        )
        
        return transition_result
    
    def _choose_professional_transition_style(self, analysis1: Dict, analysis2: Dict,
                                            mix_style: str, transition_index: int, 
                                            total_tracks: int) -> str:
        """Choose optimal transition style using AI DJ transition selector"""
        
        # Use the new intelligent transition selector
        optimal_style = self.transitions.select_optimal_transition_style(
            analysis1, analysis2, transition_index, total_tracks
        )
        
        # Apply mix style preferences as override
        if mix_style == 'party':
            # Party mixes favor energy and excitement
            if optimal_style == 'crossfade':
                # Check if we can upgrade to more exciting transition
                tempo_diff = abs(analysis1.get('tempo', 120) - analysis2.get('tempo', 120))
                if tempo_diff > 8:
                    return 'loop_roll'  # More exciting for tempo changes
                elif transition_index / total_tracks > 0.6:  # Later in mix
                    return 'filter_sweep'  # Build excitement
            return optimal_style
                
        elif mix_style == 'chill':
            # Chill mixes favor smooth transitions
            if optimal_style in ['loop_roll', 'quick_cut']:
                return 'crossfade'  # Override aggressive transitions
            return optimal_style
            
        elif mix_style == 'emotional':
            # Emotional mixes favor harmonic and meaningful transitions
            if optimal_style == 'quick_cut':
                return 'crossfade'  # More emotional
            return optimal_style
                
        else:  # journey or default
            # For journey mixes, use the AI selector with occasional overrides
            position_ratio = transition_index / (total_tracks - 1) if total_tracks > 1 else 0
            
            if position_ratio < 0.2:  # Opening - keep it smooth
                if optimal_style in ['loop_roll', 'quick_cut']:
                    return 'crossfade'
            elif position_ratio > 0.8:  # Closing - build to finale
                if optimal_style == 'crossfade':
                    return 'filter_sweep'  # More dramatic ending
            
            return optimal_style
    
    def _create_enhanced_transition(self, track1_path: str, track2_path: str,
                                  analysis1: Dict, analysis2: Dict,
                                  mix_style: str, position_in_mix: str) -> AudioSegment:
        """Create beautiful, musical transitions between tracks"""
        
        # Load audio
        audio1 = AudioSegment.from_file(track1_path)
        audio2 = AudioSegment.from_file(track2_path)
        
        # Normalize levels for consistency
        audio1 = self._normalize_for_mix(audio1)
        audio2 = self._normalize_for_mix(audio2)
        
        # Calculate compatibility for transition style decision
        compatibility = self.track_optimizer.calculate_track_compatibility(analysis1, analysis2)
        
        # Choose transition style based on compatibility and mix style
        transition_style = self._choose_transition_style(
            compatibility, mix_style, position_in_mix
        )
        
        # Find optimal mix points
        mix_out_time, mix_in_time = self._find_musical_mix_points(analysis1, analysis2)
        
        # Create the transition
        if transition_style == 'seamless':
            return self._create_seamless_transition(
                audio1, audio2, analysis1, analysis2, mix_out_time, mix_in_time
            )
        elif transition_style == 'dramatic':
            return self._create_dramatic_transition(
                audio1, audio2, analysis1, analysis2, mix_out_time, mix_in_time
            )
        elif transition_style == 'gentle':
            return self._create_gentle_transition(
                audio1, audio2, analysis1, analysis2, mix_out_time, mix_in_time
            )
        else:  # 'quick'
            return self._create_quick_transition(
                audio1, audio2, analysis1, analysis2, mix_out_time, mix_in_time
            )
    
    def _choose_transition_style(self, compatibility: Dict, mix_style: str, 
                               position_in_mix: str) -> str:
        """Choose appropriate transition style"""
        
        overall_compatibility = compatibility['overall']
        
        # High compatibility = seamless transitions
        if overall_compatibility > 0.8:
            return 'seamless'
        
        # Medium compatibility = varies by mix style
        elif overall_compatibility > 0.5:
            if mix_style == 'chill':
                return 'gentle'
            elif mix_style == 'party':
                return 'dramatic'
            else:
                return 'seamless'
        
        # Low compatibility = quick transitions or dramatic for effect
        else:
            if position_in_mix == 'end':
                return 'dramatic'  # Big finish
            else:
                return 'quick'  # Don't dwell on incompatible tracks
    
    def _find_musical_mix_points(self, analysis1: Dict, analysis2: Dict) -> Tuple[float, float]:
        """Find musically appropriate mix points"""
        
        duration1 = analysis1['duration']
        duration2 = analysis2['duration']
        
        # Default points
        mix_out_time = duration1 * 0.75
        mix_in_time = duration2 * 0.25
        
        # Try to align with beat times if available
        beat_times1 = analysis1.get('beat_times', [])
        beat_times2 = analysis2.get('beat_times', [])
        
        if len(beat_times1) > 0:
            # Find beat closest to 75% point
            target_out = duration1 * 0.75
            closest_idx = np.argmin([abs(bt - target_out) for bt in beat_times1])
            mix_out_time = beat_times1[closest_idx]
        
        if len(beat_times2) > 0:
            # Find beat closest to 25% point
            target_in = duration2 * 0.25
            closest_idx = np.argmin([abs(bt - target_in) for bt in beat_times2])
            mix_in_time = beat_times2[closest_idx]
        
        return mix_out_time, mix_in_time
    
    def _create_seamless_transition(self, audio1: AudioSegment, audio2: AudioSegment,
                                  analysis1: Dict, analysis2: Dict,
                                  mix_out_time: float, mix_in_time: float) -> AudioSegment:
        """Create perfectly seamless transition for compatible tracks"""
        
        tempo1 = analysis1['tempo']
        
        # Long, smooth transition (16-32 beats)
        transition_beats = 24
        transition_ms = int((transition_beats / tempo1) * 60 * 1000)
        
        mix_out_ms = int(mix_out_time * 1000)
        mix_in_ms = int(mix_in_time * 1000)
        
        # Extract sections
        pre_transition = audio1[:mix_out_ms]
        transition1 = audio1[mix_out_ms:mix_out_ms + transition_ms]
        transition2 = audio2[mix_in_ms:mix_in_ms + transition_ms]
        post_transition = audio2[mix_in_ms + transition_ms:]
        
        # Apply subtle EQ during transition
        transition1 = self._apply_subtle_outro_eq(transition1)
        transition2 = self._apply_subtle_intro_eq(transition2)
        
        # Create smooth crossfade
        crossfaded = AdvancedDJEffects.create_natural_crossfade(
            transition1, transition2, transition_ms
        )
        
        return pre_transition + crossfaded + post_transition
    
    def _create_dramatic_transition(self, audio1: AudioSegment, audio2: AudioSegment,
                                  analysis1: Dict, analysis2: Dict,
                                  mix_out_time: float, mix_in_time: float) -> AudioSegment:
        """Create dramatic transition with effects"""
        
        # Use shorter transition with more pronounced effects
        tempo1 = analysis1['tempo']
        transition_beats = 16
        transition_ms = int((transition_beats / tempo1) * 60 * 1000)
        
        mix_out_ms = int(mix_out_time * 1000)
        mix_in_ms = int(mix_in_time * 1000)
        
        pre_transition = audio1[:mix_out_ms]
        transition1 = audio1[mix_out_ms:mix_out_ms + transition_ms]
        transition2 = audio2[mix_in_ms:mix_in_ms + transition_ms]
        post_transition = audio2[mix_in_ms + transition_ms:]
        
        # Apply dramatic effects
        transition1 = AdvancedDJEffects._apply_dramatic_filter_sweep(transition1)
        
        # Quick, impactful crossfade
        crossfaded = AdvancedDJEffects.create_natural_crossfade(
            transition1, transition2, transition_ms // 2
        )
        
        return pre_transition + crossfaded + post_transition
    
    def _create_gentle_transition(self, audio1: AudioSegment, audio2: AudioSegment,
                                analysis1: Dict, analysis2: Dict,
                                mix_out_time: float, mix_in_time: float) -> AudioSegment:
        """Create very gentle, subtle transition"""
        
        # Very long transition for smooth flow
        tempo1 = analysis1['tempo']
        transition_beats = 32
        transition_ms = int((transition_beats / tempo1) * 60 * 1000)
        
        mix_out_ms = int(mix_out_time * 1000)
        mix_in_ms = int(mix_in_time * 1000)
        
        pre_transition = audio1[:mix_out_ms]
        transition1 = audio1[mix_out_ms:mix_out_ms + transition_ms]
        transition2 = audio2[mix_in_ms:mix_in_ms + transition_ms]
        post_transition = audio2[mix_in_ms + transition_ms:]
        
        # Very subtle EQ changes
        transition1 = self._apply_gentle_outro_eq(transition1)
        transition2 = self._apply_gentle_intro_eq(transition2)
        
        # Very gradual crossfade
        crossfaded = AdvancedDJEffects.create_natural_crossfade(
            transition1, transition2, transition_ms
        )
        
        return pre_transition + crossfaded + post_transition
    
    def _create_quick_transition(self, audio1: AudioSegment, audio2: AudioSegment,
                               analysis1: Dict, analysis2: Dict,
                               mix_out_time: float, mix_in_time: float) -> AudioSegment:
        """Create quick transition for incompatible tracks"""
        
        # Very short transition
        tempo1 = analysis1['tempo']
        transition_beats = 4
        transition_ms = int((transition_beats / tempo1) * 60 * 1000)
        
        mix_out_ms = int(mix_out_time * 1000)
        mix_in_ms = int(mix_in_time * 1000)
        
        pre_transition = audio1[:mix_out_ms]
        post_transition = audio2[mix_in_ms:]
        
        # Very quick crossfade to minimize incompatibility
        if transition_ms > 0:
            fade_out = audio1[mix_out_ms:mix_out_ms + transition_ms].fade_out(transition_ms)
            fade_in = audio2[mix_in_ms:mix_in_ms + transition_ms].fade_in(transition_ms)
            quick_transition = fade_out.overlay(fade_in)
            
            return pre_transition + quick_transition + post_transition
        else:
            return pre_transition + post_transition
    
    def _normalize_for_mix(self, audio: AudioSegment, target_lufs: float = -18.0) -> AudioSegment:
        """Normalize audio for consistent mix levels"""
        
        # Simple RMS normalization
        current_rms = audio.rms
        if current_rms > 0:
            target_rms = 10 ** (target_lufs / 20) * 32767
            gain_adjustment = 20 * np.log10(target_rms / current_rms)
            return audio + gain_adjustment
        return audio
    
    def _apply_subtle_outro_eq(self, audio: AudioSegment) -> AudioSegment:
        """Apply subtle EQ for outro"""
        try:
            from pydub.effects import high_pass_filter
            return high_pass_filter(audio, 100)  # Very gentle high-pass
        except:
            return audio
    
    def _apply_subtle_intro_eq(self, audio: AudioSegment) -> AudioSegment:
        """Apply subtle EQ for intro"""
        try:
            from pydub.effects import low_pass_filter
            return low_pass_filter(audio, 12000)  # Very gentle low-pass
        except:
            return audio
    
    def _apply_gentle_outro_eq(self, audio: AudioSegment) -> AudioSegment:
        """Apply very gentle EQ for chill outros"""
        try:
            from pydub.effects import high_pass_filter
            return high_pass_filter(audio, 80)  # Even more gentle
        except:
            return audio
    
    def _apply_gentle_intro_eq(self, audio: AudioSegment) -> AudioSegment:
        """Apply very gentle EQ for chill intros"""
        try:
            from pydub.effects import low_pass_filter
            return low_pass_filter(audio, 15000)  # Very subtle
        except:
            return audio
    
    def _apply_mix_mastering(self, mixed_audio: AudioSegment) -> AudioSegment:
        """Apply professional mastering touches to the complete mix"""
        
        logger.info("Applying professional mastering chain")
        
        # First normalize for consistent level
        normalized = self._normalize_for_mix(mixed_audio, target_lufs=-16.0)
        
        # Apply professional mastering chain
        mastered = self.production.apply_professional_mastering(normalized, 'journey')
        
        logger.info("Applied professional mastering to complete mix")
        return mastered
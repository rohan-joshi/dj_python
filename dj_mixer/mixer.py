import librosa
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pydub import AudioSegment
from .audio_analyzer import AudioAnalyzer
from .effects import AudioEffects
from .utils import AudioUtils
from .harmonic_analyzer import HarmonicAnalyzer
from .cohesive_mix_builder import CohesiveMixBuilder

logger = logging.getLogger(__name__)

class DJMixer:
    def __init__(self, crossfade_duration_beats: int = 8, tempo_tolerance: float = 0.2, 
                 auto_optimize: bool = True, professional_mode: bool = True,
                 mix_style: str = 'journey'):
        self.analyzer = AudioAnalyzer()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.cohesive_mix_builder = CohesiveMixBuilder()
        self.crossfade_duration_beats = crossfade_duration_beats
        self.tempo_tolerance = tempo_tolerance
        self.auto_optimize = auto_optimize
        self.professional_mode = professional_mode
        self.mix_style = mix_style
        self.track_analyses = {}
    
    def analyze_tracks(self, file_paths: List[str]) -> Dict[str, dict]:
        analyses = {}
        
        for file_path in file_paths:
            try:
                if not AudioUtils.validate_audio_file(file_path):
                    logger.warning(f"Skipping invalid audio file: {file_path}")
                    continue
                
                analysis = self.analyzer.analyze_track(file_path)
                analyses[file_path] = analysis
                
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                continue
        
        self.track_analyses = analyses
        return analyses
    
    def calculate_tempo_adjustment(self, tempo1: float, tempo2: float) -> Tuple[float, bool]:
        ratio = tempo2 / tempo1
        
        if abs(1 - ratio) > self.tempo_tolerance:
            logger.warning(f"Tempo difference too large: {tempo1:.1f} BPM vs {tempo2:.1f} BPM")
            return 1.0, False
        
        return ratio, True
    
    def time_stretch_audio(self, file_path: str, stretch_ratio: float) -> AudioSegment:
        try:
            y, sr = self.analyzer.load_audio(file_path)
            
            if abs(1 - stretch_ratio) < 0.01:
                return AudioUtils.load_audio_segment(file_path)
            
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_ratio)
            
            audio_segment = AudioUtils.load_audio_segment(file_path)
            
            stretched_samples = (y_stretched * 32767).astype(np.int16)
            
            if audio_segment.channels == 2:
                stretched_samples = np.repeat(stretched_samples.reshape(-1, 1), 2, axis=1)
                stretched_samples = stretched_samples.flatten()
            
            stretched_audio = AudioSegment(
                stretched_samples.tobytes(),
                frame_rate=audio_segment.frame_rate,
                sample_width=audio_segment.sample_width,
                channels=audio_segment.channels
            )
            
            logger.debug(f"Time-stretched audio by factor {stretch_ratio:.3f}")
            return stretched_audio
            
        except Exception as e:
            logger.error(f"Error time-stretching audio: {e}")
            return AudioUtils.load_audio_segment(file_path)
    
    def find_optimal_mix_points(self, analysis1: dict, analysis2: dict) -> Tuple[float, float]:
        try:
            duration1 = analysis1['duration']
            duration2 = analysis2['duration']
            
            mix_out_time = max(duration1 * 0.75, duration1 - 30)
            mix_in_time = min(duration2 * 0.25, 30)
            
            beat_times1 = analysis1['beat_times']
            beat_times2 = analysis2['beat_times']
            
            if len(beat_times1) > 0:
                beat_idx = np.argmin(np.abs(beat_times1 - mix_out_time))
                mix_out_time = beat_times1[beat_idx]
            
            if len(beat_times2) > 0:
                beat_idx = np.argmin(np.abs(beat_times2 - mix_in_time))
                mix_in_time = beat_times2[beat_idx]
            
            return mix_out_time, mix_in_time
            
        except Exception as e:
            logger.error(f"Error finding optimal mix points: {e}")
            return analysis1['duration'] * 0.75, analysis2['duration'] * 0.25
    
    def prepare_track_for_mixing(self, file_path: str, analysis: dict, 
                                stretch_ratio: float = 1.0) -> AudioSegment:
        try:
            if abs(1 - stretch_ratio) > 0.01:
                audio = self.time_stretch_audio(file_path, stretch_ratio)
            else:
                audio = AudioUtils.load_audio_segment(file_path)
            
            audio = AudioUtils.convert_to_stereo(audio)
            audio = AudioUtils.match_sample_rate(audio, 44100)
            audio = AudioEffects.normalize_levels(audio, target_dbfs=-20.0)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error preparing track {file_path}: {e}")
            return AudioUtils.load_audio_segment(file_path)
    
    def mix_two_tracks(self, track1_path: str, track2_path: str, 
                      analysis1: dict, analysis2: dict) -> AudioSegment:
        try:
            if self.professional_mode:
                # Use professional DJ mixing techniques
                mixing_style = self.true_dj_mixer.choose_mixing_style(analysis1, analysis2)
                logger.info(f"Using professional DJ technique: {mixing_style}")
                
                result = self.true_dj_mixer.mix_like_real_dj(
                    track1_path, track2_path, analysis1, analysis2, mixing_style
                )
                
                logger.info(f"Professional DJ mix completed: {track1_path} -> {track2_path}")
                return result
            
            else:
                # Original automated mixing approach
                tempo1 = analysis1['tempo']
                tempo2 = analysis2['tempo']
                
                stretch_ratio, can_match = self.calculate_tempo_adjustment(tempo1, tempo2)
                
                if not can_match:
                    logger.warning(f"Cannot match tempos, mixing without tempo adjustment")
                    stretch_ratio = 1.0
                
                audio1 = self.prepare_track_for_mixing(track1_path, analysis1)
                audio2 = self.prepare_track_for_mixing(track2_path, analysis2, stretch_ratio)
                
                y1 = analysis1.get('audio_data')
                y2 = analysis2.get('audio_data')
                
                if y1 is not None and y2 is not None:
                    mix_points = self.mix_point_detector.find_optimal_mix_points(
                        analysis1, analysis2, y1, y2, analysis1['sample_rate']
                    )
                    mix_out_time = mix_points['mix_out_time']
                    mix_in_time = mix_points['mix_in_time']
                    logger.info(f"Using advanced mix points: out={mix_out_time:.1f}s, in={mix_in_time:.1f}s")
                else:
                    mix_out_time, mix_in_time = self.find_optimal_mix_points(analysis1, analysis2)
                
                compatibility = self.track_optimizer.calculate_track_compatibility(analysis1, analysis2)
                crossfade_duration_beats = self._calculate_dynamic_crossfade_duration(
                    compatibility, analysis1, analysis2
                )
                
                crossfade_duration_ms = int(crossfade_duration_beats * (60 / tempo1) * 1000)
                
                mix_out_ms = int(mix_out_time * 1000)
                mix_in_ms = int(mix_in_time * 1000)
                
                pre_mix = audio1[:mix_out_ms]
                
                track1_tail = audio1[mix_out_ms:]
                track2_intro = audio2[:mix_in_ms + crossfade_duration_ms]
                post_mix = audio2[mix_in_ms + crossfade_duration_ms:]
                
                track1_tail, track2_intro = self._apply_intelligent_effects(
                    track1_tail, track2_intro, analysis1, analysis2, compatibility
                )
                
                if 'beat_times' in analysis1 and 'beat_times' in analysis2:
                    crossfaded = AudioEffects.create_beat_aligned_crossfade(
                        track1_tail, track2_intro,
                        analysis1['beat_times'], analysis2['beat_times'],
                        crossfade_duration_beats
                    )
                else:
                    crossfaded = AudioEffects.equal_power_crossfade(
                        track1_tail, track2_intro, crossfade_duration_ms
                    )
                
                result = pre_mix + crossfaded + post_mix
                
                logger.info(f"Mixed {track1_path} -> {track2_path} with {crossfade_duration_ms}ms crossfade (compatibility: {compatibility['overall']:.2f})")
                return result
            
        except Exception as e:
            logger.error(f"Error mixing tracks: {e}")
            audio1 = AudioUtils.load_audio_segment(track1_path)
            audio2 = AudioUtils.load_audio_segment(track2_path)
            return audio1 + audio2
    
    def create_dj_mix(self, file_paths: List[str]) -> AudioSegment:
        if len(file_paths) == 0:
            raise ValueError("No audio files provided")
        
        if len(file_paths) == 1:
            return AudioUtils.load_audio_segment(file_paths[0])
        
        logger.info(f"Creating beautiful DJ mix from {len(file_paths)} tracks")
        
        analyses = self.analyze_tracks(file_paths)
        
        valid_tracks = [path for path in file_paths if path in analyses]
        
        if len(valid_tracks) == 0:
            raise ValueError("No valid audio files found")
        
        if len(valid_tracks) == 1:
            return AudioUtils.load_audio_segment(valid_tracks[0])
        
        # Use the cohesive mix builder for beautiful studio-quality mixes
        logger.info(f"Building cohesive {self.mix_style} mix...")
        
        # Filter analyses to only include valid tracks
        valid_analyses = {path: analyses[path] for path in valid_tracks}
        
        # Create the beautiful mix
        mixed_audio = self.cohesive_mix_builder.create_beautiful_mix(
            valid_analyses, self.mix_style
        )
        
        logger.info("Beautiful DJ mix creation completed")
        return mixed_audio
    
    def get_mix_statistics(self) -> dict:
        if not self.track_analyses:
            return {}
        
        tempos = [analysis['tempo'] for analysis in self.track_analyses.values()]
        durations = [analysis['duration'] for analysis in self.track_analyses.values()]
        
        stats = {
            'total_tracks': len(self.track_analyses),
            'avg_tempo': np.mean(tempos),
            'tempo_range': (min(tempos), max(tempos)),
            'total_duration': sum(durations),
            'avg_track_duration': np.mean(durations)
        }
        
        return stats
    
    def _calculate_dynamic_crossfade_duration(self, compatibility: Dict, 
                                            analysis1: Dict, analysis2: Dict) -> int:
        base_duration = self.crossfade_duration_beats
        
        if compatibility['overall'] > 0.8:
            return max(4, base_duration - 2)
        elif compatibility['overall'] > 0.6:
            return base_duration
        elif compatibility['overall'] > 0.4:
            return base_duration + 2
        else:
            return min(16, base_duration + 4)
    
    def _apply_intelligent_effects(self, track1_tail: AudioSegment, track2_intro: AudioSegment,
                                 analysis1: Dict, analysis2: Dict, compatibility: Dict) -> Tuple[AudioSegment, AudioSegment]:
        
        if compatibility['key'] > 0.8:
            track1_tail = AudioEffects.apply_high_pass_filter(track1_tail, 150)
            track2_intro = AudioEffects.apply_low_pass_filter(track2_intro, 10000)
        elif compatibility['key'] > 0.5:
            track1_tail = AudioEffects.apply_high_pass_filter(track1_tail, 200)
            track2_intro = AudioEffects.apply_low_pass_filter(track2_intro, 8000)
        else:
            track1_tail = AudioEffects.apply_high_pass_filter(track1_tail, 300)
            track2_intro = AudioEffects.apply_low_pass_filter(track2_intro, 6000)
            
            track1_tail = AudioEffects.apply_reverb_tail(track1_tail, 3000, 0.4)
        
        if compatibility['energy'] < 0.5:
            energy1 = np.mean(analysis1.get('energy', [0.5]))
            energy2 = np.mean(analysis2.get('energy', [0.5]))
            
            if energy2 > energy1:
                track2_intro = AudioEffects.apply_filter_sweep(
                    track2_intro, start_freq=8000, end_freq=20000, sweep_duration_ms=2000
                )
            else:
                track1_tail = AudioEffects.apply_filter_sweep(
                    track1_tail, start_freq=20000, end_freq=1000, sweep_duration_ms=3000
                )
        
        return track1_tail, track2_intro
import numpy as np
import logging
from pydub import AudioSegment
from pydub.effects import normalize, high_pass_filter, low_pass_filter
from scipy import signal
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class AudioEffects:
    def __init__(self):
        pass
    
    @staticmethod
    def equal_power_crossfade(audio1: AudioSegment, audio2: AudioSegment, 
                             crossfade_duration_ms: int) -> AudioSegment:
        try:
            if crossfade_duration_ms <= 0:
                return audio1 + audio2
            
            fade_duration = min(crossfade_duration_ms, len(audio1), len(audio2))
            
            fade_out_audio = audio1[-fade_duration:]
            fade_in_audio = audio2[:fade_duration]
            
            overlap_samples = fade_duration
            
            fade_out_curve = AudioEffects._generate_equal_power_fade_out(overlap_samples)
            fade_in_curve = AudioEffects._generate_equal_power_fade_in(overlap_samples)
            
            faded_out = AudioEffects._apply_fade_curve(fade_out_audio, fade_out_curve)
            faded_in = AudioEffects._apply_fade_curve(fade_in_audio, fade_in_curve)
            
            crossfaded_section = faded_out.overlay(faded_in)
            
            result = audio1[:-fade_duration] + crossfaded_section + audio2[fade_duration:]
            
            logger.debug(f"Applied equal power crossfade: {fade_duration}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error applying crossfade: {e}")
            return audio1 + audio2
    
    @staticmethod
    def _generate_equal_power_fade_out(samples: int) -> np.ndarray:
        t = np.linspace(0, np.pi/2, samples)
        return np.cos(t)
    
    @staticmethod
    def _generate_equal_power_fade_in(samples: int) -> np.ndarray:
        t = np.linspace(0, np.pi/2, samples)
        return np.sin(t)
    
    @staticmethod
    def _apply_fade_curve(audio: AudioSegment, curve: np.ndarray) -> AudioSegment:
        samples = audio.get_array_of_samples()
        
        if audio.channels == 2:
            samples_left = samples[0::2]
            samples_right = samples[1::2]
            
            curve_int = (curve * 32767).astype(np.int16)
            
            faded_left = (samples_left * curve_int / 32767).astype(np.int16)
            faded_right = (samples_right * curve_int / 32767).astype(np.int16)
            
            interleaved = np.empty(len(samples), dtype=np.int16)
            interleaved[0::2] = faded_left
            interleaved[1::2] = faded_right
            
            result_samples = interleaved
        else:
            curve_int = (curve * 32767).astype(np.int16)
            result_samples = (samples * curve_int / 32767).astype(np.int16)
        
        return audio._spawn(result_samples.tobytes())
    
    @staticmethod
    def apply_high_pass_filter(audio: AudioSegment, cutoff_freq: int = 200) -> AudioSegment:
        try:
            filtered = high_pass_filter(audio, cutoff_freq)
            logger.debug(f"Applied high-pass filter at {cutoff_freq}Hz")
            return filtered
        except Exception as e:
            logger.error(f"Error applying high-pass filter: {e}")
            return audio
    
    @staticmethod
    def apply_low_pass_filter(audio: AudioSegment, cutoff_freq: int = 8000) -> AudioSegment:
        try:
            filtered = low_pass_filter(audio, cutoff_freq)
            logger.debug(f"Applied low-pass filter at {cutoff_freq}Hz")
            return filtered
        except Exception as e:
            logger.error(f"Error applying low-pass filter: {e}")
            return audio
    
    @staticmethod
    def apply_reverb_tail(audio: AudioSegment, decay_time_ms: int = 2000, 
                         wet_level: float = 0.3) -> AudioSegment:
        try:
            reverb_duration = min(decay_time_ms, 5000)
            
            decay_curve = np.exp(-np.linspace(0, 5, reverb_duration * audio.frame_rate // 1000))
            
            silence = AudioSegment.silent(duration=reverb_duration, frame_rate=audio.frame_rate)
            
            reverb_tail = AudioEffects._apply_fade_curve(
                audio[-reverb_duration:] + silence, 
                decay_curve
            )
            
            wet_reverb = reverb_tail - (20 - int(20 * wet_level))
            
            result = audio + wet_reverb
            
            logger.debug(f"Applied reverb tail: {decay_time_ms}ms, wet level: {wet_level}")
            return result
            
        except Exception as e:
            logger.error(f"Error applying reverb tail: {e}")
            return audio
    
    @staticmethod
    def apply_filter_sweep(audio: AudioSegment, start_freq: int = 20000, 
                          end_freq: int = 200, sweep_duration_ms: int = 4000) -> AudioSegment:
        try:
            if sweep_duration_ms > len(audio):
                sweep_duration_ms = len(audio)
            
            sweep_samples = sweep_duration_ms * audio.frame_rate // 1000
            
            frequencies = np.logspace(
                np.log10(start_freq), 
                np.log10(end_freq), 
                sweep_samples
            )
            
            filtered_audio = audio
            
            for i, freq in enumerate(frequencies):
                start_sample = i * len(audio) // len(frequencies)
                end_sample = (i + 1) * len(audio) // len(frequencies)
                
                segment = audio[start_sample:end_sample]
                if freq < 1000:
                    segment = AudioEffects.apply_low_pass_filter(segment, int(freq))
                else:
                    segment = AudioEffects.apply_high_pass_filter(segment, int(freq))
                
                if i == 0:
                    filtered_audio = segment
                else:
                    filtered_audio += segment
            
            logger.debug(f"Applied filter sweep: {start_freq}Hz -> {end_freq}Hz over {sweep_duration_ms}ms")
            return filtered_audio
            
        except Exception as e:
            logger.error(f"Error applying filter sweep: {e}")
            return audio
    
    @staticmethod
    def normalize_levels(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
        try:
            normalized = normalize(audio)
            
            current_dbfs = normalized.dBFS
            adjustment = target_dbfs - current_dbfs
            
            result = normalized + adjustment
            
            logger.debug(f"Normalized audio to {target_dbfs} dBFS")
            return result
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio
    
    @staticmethod
    def apply_gain_automation(audio: AudioSegment, gain_curve: np.ndarray) -> AudioSegment:
        try:
            if len(gain_curve) != len(audio):
                gain_curve = np.interp(
                    np.linspace(0, 1, len(audio)),
                    np.linspace(0, 1, len(gain_curve)),
                    gain_curve
                )
            
            result = AudioEffects._apply_fade_curve(audio, gain_curve)
            
            logger.debug("Applied gain automation curve")
            return result
            
        except Exception as e:
            logger.error(f"Error applying gain automation: {e}")
            return audio
    
    @staticmethod
    def create_beat_aligned_crossfade(audio1: AudioSegment, audio2: AudioSegment,
                                    beat_positions_1: np.ndarray, beat_positions_2: np.ndarray,
                                    crossfade_beats: int = 8) -> AudioSegment:
        try:
            if len(beat_positions_1) < crossfade_beats or len(beat_positions_2) < crossfade_beats:
                logger.warning("Not enough beats for beat-aligned crossfade, using regular crossfade")
                return AudioEffects.equal_power_crossfade(audio1, audio2, 8000)
            
            beat_duration_1 = np.mean(np.diff(beat_positions_1[-crossfade_beats:]))
            crossfade_duration_ms = int(beat_duration_1 * crossfade_beats * 1000)
            
            crossfade_start_1 = beat_positions_1[-crossfade_beats]
            crossfade_start_ms_1 = int(crossfade_start_1 * 1000)
            
            crossfade_start_2 = beat_positions_2[crossfade_beats - 1]
            crossfade_start_ms_2 = int(crossfade_start_2 * 1000)
            
            pre_crossfade_1 = audio1[:crossfade_start_ms_1]
            crossfade_section_1 = audio1[crossfade_start_ms_1:crossfade_start_ms_1 + crossfade_duration_ms]
            
            crossfade_section_2 = audio2[crossfade_start_ms_2:crossfade_start_ms_2 + crossfade_duration_ms]
            post_crossfade_2 = audio2[crossfade_start_ms_2 + crossfade_duration_ms:]
            
            crossfaded_section = AudioEffects.equal_power_crossfade(
                crossfade_section_1, 
                crossfade_section_2, 
                crossfade_duration_ms
            )
            
            result = pre_crossfade_1 + crossfaded_section + post_crossfade_2
            
            logger.debug(f"Applied beat-aligned crossfade: {crossfade_beats} beats, {crossfade_duration_ms}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error applying beat-aligned crossfade: {e}")
            return AudioEffects.equal_power_crossfade(audio1, audio2, 8000)
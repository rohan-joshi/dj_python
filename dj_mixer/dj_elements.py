#!/usr/bin/env python3

import numpy as np
import random
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DJElements:
    """
    Professional DJ elements: drops, air horns, effects, vocal samples
    """
    
    def __init__(self):
        self.sound_cache = {}
        self._generate_signature_sounds()
        
    def _generate_signature_sounds(self):
        """Generate signature DJ sounds"""
        logger.debug("Generating signature DJ sounds")
        
        # Air horn
        self.sound_cache['air_horn'] = self._create_air_horn()
        
        # Siren
        self.sound_cache['siren'] = self._create_siren()
        
        # Vinyl scratch
        self.sound_cache['scratch'] = self._create_scratch_sound()
        
        # White noise sweep
        self.sound_cache['noise_sweep'] = self._create_noise_sweep()
        
        # Reverse crash
        self.sound_cache['reverse_crash'] = self._create_reverse_crash()
        
        # Vocal drops (synthesized)
        self.sound_cache['vocal_drop'] = self._create_vocal_drop()
    
    def _create_air_horn(self, duration_ms: int = 800) -> AudioSegment:
        """Create classic air horn sound"""
        # Create harmonically rich sound typical of air horns
        fundamental = 200  # Hz
        
        # Generate multiple harmonics
        air_horn = Sine(fundamental).to_audio_segment(duration=duration_ms)
        air_horn += Sine(fundamental * 1.5).to_audio_segment(duration=duration_ms) - 3
        air_horn += Sine(fundamental * 2).to_audio_segment(duration=duration_ms) - 6
        air_horn += Sine(fundamental * 2.5).to_audio_segment(duration=duration_ms) - 9
        
        # Add envelope (attack and decay)
        air_horn = self._apply_envelope(air_horn, attack_ms=50, decay_ms=200)
        
        # Add some distortion/saturation
        air_horn = air_horn + 6  # Boost for saturation effect
        
        return air_horn.normalize()
    
    def _create_siren(self, duration_ms: int = 2000) -> AudioSegment:
        """Create siren sweep sound"""
        sample_rate = 44100
        samples = int(sample_rate * duration_ms / 1000)
        
        # Create frequency sweep from 800Hz to 1200Hz and back
        t = np.linspace(0, duration_ms / 1000, samples)
        freq_sweep = 800 + 400 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz modulation
        
        # Generate the sine wave with frequency modulation
        phase = np.cumsum(2 * np.pi * freq_sweep / sample_rate)
        siren_wave = np.sin(phase)
        
        # Apply envelope
        envelope = np.exp(-t * 0.5)  # Exponential decay
        siren_wave *= envelope
        
        # Convert to AudioSegment
        siren_wave = (siren_wave * 32767).astype(np.int16)
        siren = AudioSegment(
            siren_wave.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        return siren.normalize()
    
    def _create_scratch_sound(self, duration_ms: int = 500) -> AudioSegment:
        """Create vinyl scratch sound effect"""
        # Use filtered white noise to simulate scratch
        noise = WhiteNoise().to_audio_segment(duration=duration_ms)
        
        # Apply multiple filters to shape the sound
        scratch = noise.high_pass_filter(1000)  # Remove low frequencies
        scratch = scratch.low_pass_filter(8000)  # Remove very high frequencies
        
        # Add amplitude modulation for scratch rhythm
        samples = scratch.get_array_of_samples()
        samples = np.array(samples).astype(np.float32)
        
        # Create scratch pattern
        t = np.linspace(0, duration_ms / 1000, len(samples))
        modulation = np.abs(np.sin(2 * np.pi * 12 * t))  # 12 Hz modulation
        samples *= modulation
        
        samples = samples.astype(np.int16)
        scratch = AudioSegment(
            samples.tobytes(),
            frame_rate=scratch.frame_rate,
            sample_width=scratch.sample_width,
            channels=scratch.channels
        )
        
        return scratch.normalize() - 6  # Reduce volume
    
    def _create_noise_sweep(self, duration_ms: int = 1000) -> AudioSegment:
        """Create uplifting white noise sweep"""
        noise = WhiteNoise().to_audio_segment(duration=duration_ms)
        
        # Create frequency sweep using high-pass filter
        samples = noise.get_array_of_samples()
        samples = np.array(samples).astype(np.float32)
        
        # Apply envelope for sweep effect
        t = np.linspace(0, 1, len(samples))
        envelope = t ** 2  # Quadratic rise
        samples *= envelope
        
        samples = samples.astype(np.int16)
        sweep = AudioSegment(
            samples.tobytes(),
            frame_rate=noise.frame_rate,
            sample_width=noise.sample_width,
            channels=noise.channels
        )
        
        # Apply high-pass filter sweep
        sweep = sweep.high_pass_filter(200)
        
        return sweep.normalize() - 3
    
    def _create_reverse_crash(self, duration_ms: int = 1500) -> AudioSegment:
        """Create reverse crash/cymbal effect"""
        # Generate crash-like sound using multiple frequencies
        crash_freqs = [3000, 5000, 7000, 9000, 11000]
        crash = AudioSegment.silent(duration=duration_ms)
        
        for freq in crash_freqs:
            component = Sine(freq).to_audio_segment(duration=duration_ms)
            # Add some randomness to make it more crash-like
            component = component + random.randint(-12, -6)
            crash = crash.overlay(component)
        
        # Add noise component
        noise = WhiteNoise().to_audio_segment(duration=duration_ms)
        noise = noise.high_pass_filter(2000) - 15
        crash = crash.overlay(noise)
        
        # Reverse the crash
        crash = crash.reverse()
        
        # Apply reverse envelope (builds up)
        samples = crash.get_array_of_samples()
        samples = np.array(samples).astype(np.float32)
        
        t = np.linspace(0, 1, len(samples))
        envelope = t ** 3  # Cubic rise for dramatic buildup
        samples *= envelope
        
        samples = samples.astype(np.int16)
        reverse_crash = AudioSegment(
            samples.tobytes(),
            frame_rate=crash.frame_rate,
            sample_width=crash.sample_width,
            channels=crash.channels
        )
        
        return reverse_crash.normalize() - 6
    
    def _create_vocal_drop(self, duration_ms: int = 600) -> AudioSegment:
        """Create synthesized vocal drop effect"""
        # Create a formant-like sound that resembles vocal drops
        fundamental = 150  # Hz
        
        # Create formants typical of vocal sounds
        formant1 = Sine(fundamental).to_audio_segment(duration=duration_ms)
        formant2 = Sine(fundamental * 3).to_audio_segment(duration=duration_ms) - 6
        formant3 = Sine(fundamental * 5).to_audio_segment(duration=duration_ms) - 12
        
        vocal = formant1.overlay(formant2).overlay(formant3)
        
        # Apply vocal-like envelope
        vocal = self._apply_envelope(vocal, attack_ms=20, decay_ms=400)
        
        # Add some noise for realism
        noise = WhiteNoise().to_audio_segment(duration=duration_ms)
        noise = noise.high_pass_filter(2000) - 20
        vocal = vocal.overlay(noise)
        
        return vocal.normalize() - 3
    
    def _apply_envelope(self, audio: AudioSegment, 
                       attack_ms: int = 50, decay_ms: int = 200) -> AudioSegment:
        """Apply attack/decay envelope to audio"""
        samples = audio.get_array_of_samples()
        samples = np.array(samples).astype(np.float32)
        
        duration_ms = len(audio)
        attack_samples = int(len(samples) * attack_ms / duration_ms)
        decay_samples = int(len(samples) * decay_ms / duration_ms)
        
        # Create envelope
        envelope = np.ones(len(samples))
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase (from the end)
        if decay_samples > 0:
            decay_start = len(samples) - decay_samples
            envelope[decay_start:] = np.linspace(1, 0, decay_samples)
        
        samples *= envelope
        samples = samples.astype(np.int16)
        
        return AudioSegment(
            samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
    
    def add_air_horn(self, audio: AudioSegment, position_ms: int, 
                    volume_offset: int = -6) -> AudioSegment:
        """Add air horn at specific position"""
        logger.debug(f"Adding air horn at {position_ms}ms")
        
        air_horn = self.sound_cache['air_horn'] + volume_offset
        return audio.overlay(air_horn, position=position_ms)
    
    def add_siren(self, audio: AudioSegment, position_ms: int,
                 volume_offset: int = -9) -> AudioSegment:
        """Add siren at specific position"""
        logger.debug(f"Adding siren at {position_ms}ms")
        
        siren = self.sound_cache['siren'] + volume_offset
        return audio.overlay(siren, position=position_ms)
    
    def add_scratch_effect(self, audio: AudioSegment, position_ms: int,
                          volume_offset: int = -12) -> AudioSegment:
        """Add scratch effect at specific position"""
        logger.debug(f"Adding scratch effect at {position_ms}ms")
        
        scratch = self.sound_cache['scratch'] + volume_offset
        return audio.overlay(scratch, position=position_ms)
    
    def add_noise_sweep(self, audio: AudioSegment, position_ms: int,
                       volume_offset: int = -9) -> AudioSegment:
        """Add noise sweep for build-ups"""
        logger.debug(f"Adding noise sweep at {position_ms}ms")
        
        sweep = self.sound_cache['noise_sweep'] + volume_offset
        return audio.overlay(sweep, position=position_ms)
    
    def add_reverse_crash(self, audio: AudioSegment, position_ms: int,
                         volume_offset: int = -6) -> AudioSegment:
        """Add reverse crash before drops"""
        logger.debug(f"Adding reverse crash at {position_ms}ms")
        
        crash = self.sound_cache['reverse_crash'] + volume_offset
        return audio.overlay(crash, position=position_ms)
    
    def add_vocal_drop(self, audio: AudioSegment, position_ms: int,
                      volume_offset: int = -9) -> AudioSegment:
        """Add vocal drop effect"""
        logger.debug(f"Adding vocal drop at {position_ms}ms")
        
        vocal = self.sound_cache['vocal_drop'] + volume_offset
        return audio.overlay(vocal, position=position_ms)
    
    def create_backspin_effect(self, audio: AudioSegment, 
                              position_ms: int, duration_ms: int = 1000) -> AudioSegment:
        """Create backspin effect for dramatic transitions"""
        logger.debug(f"Creating backspin at {position_ms}ms")
        
        # Extract section for backspin
        backspin_start = max(0, position_ms - duration_ms)
        backspin_section = audio[backspin_start:position_ms]
        
        # Reverse the section
        reversed_section = backspin_section.reverse()
        
        # Apply pitch bend effect (simulate turntable slowdown)
        samples = reversed_section.get_array_of_samples()
        samples = np.array(samples).astype(np.float32)
        
        # Create pitch bend (speed reduction over time)
        t = np.linspace(1, 0.3, len(samples))  # From normal speed to 30% speed
        
        # Simple pitch effect by time-stretching simulation
        # This is a simplified version - real pitch bending is more complex
        stretched_samples = []
        for i, speed in enumerate(t):
            if i < len(samples):
                stretched_samples.append(samples[i] * speed)
        
        # Pad or truncate to original length
        if len(stretched_samples) < len(samples):
            stretched_samples.extend([0] * (len(samples) - len(stretched_samples)))
        else:
            stretched_samples = stretched_samples[:len(samples)]
        
        stretched_samples = np.array(stretched_samples).astype(np.int16)
        
        backspin_audio = AudioSegment(
            stretched_samples.tobytes(),
            frame_rate=reversed_section.frame_rate,
            sample_width=reversed_section.sample_width,
            channels=reversed_section.channels
        )
        
        # Replace the original section with backspin
        before = audio[:backspin_start]
        after = audio[position_ms:]
        
        return before + backspin_audio + after
    
    def add_quick_cut_effect(self, audio: AudioSegment, 
                           cut_positions: List[int], 
                           cut_duration_ms: int = 100) -> AudioSegment:
        """Add quick cuts for surprise factor"""
        logger.debug(f"Adding quick cuts at {len(cut_positions)} positions")
        
        # Sort positions in reverse order to maintain indices
        cut_positions = sorted(cut_positions, reverse=True)
        
        for position in cut_positions:
            if position + cut_duration_ms < len(audio):
                # Create a brief silence
                before = audio[:position]
                after = audio[position + cut_duration_ms:]
                silence = AudioSegment.silent(duration=cut_duration_ms // 4)  # Shorter silence
                audio = before + silence + after
        
        return audio
    
    def add_smart_effects(self, audio: AudioSegment, 
                         track_analyses: List[Dict],
                         mix_style: str) -> AudioSegment:
        """Intelligently add effects based on mix style and track analysis"""
        logger.info(f"Adding smart DJ effects for {mix_style} mix")
        
        total_duration = len(audio)
        effects_added = 0
        
        # Define effect probabilities by mix style
        effect_chances = {
            'party': {
                'air_horn': 0.4,
                'siren': 0.2,
                'vocal_drop': 0.3,
                'noise_sweep': 0.5,
                'reverse_crash': 0.3
            },
            'journey': {
                'air_horn': 0.1,
                'siren': 0.05,
                'vocal_drop': 0.1,
                'noise_sweep': 0.3,
                'reverse_crash': 0.2
            },
            'chill': {
                'air_horn': 0.05,
                'siren': 0.02,
                'vocal_drop': 0.05,
                'noise_sweep': 0.1,
                'reverse_crash': 0.1
            },
            'emotional': {
                'air_horn': 0.02,
                'siren': 0.01,
                'vocal_drop': 0.05,
                'noise_sweep': 0.15,
                'reverse_crash': 0.15
            }
        }
        
        chances = effect_chances.get(mix_style, effect_chances['journey'])
        
        # Add effects at strategic points
        for analysis in track_analyses:
            track_start = analysis.get('mix_start_ms', 0)
            track_energy = analysis.get('energy', 0.5)
            
            # Handle energy arrays
            if isinstance(track_energy, np.ndarray):
                track_energy = np.mean(track_energy)
            
            # Higher energy tracks get more effects
            energy_multiplier = track_energy * 1.5
            
            # Air horns at high energy moments
            if random.random() < chances['air_horn'] * energy_multiplier:
                effect_pos = track_start + random.randint(8000, 24000)
                if effect_pos < total_duration:
                    audio = self.add_air_horn(audio, effect_pos)
                    effects_added += 1
            
            # Noise sweeps before drops/builds
            if random.random() < chances['noise_sweep'] * energy_multiplier:
                effect_pos = track_start + random.randint(16000, 32000)
                if effect_pos < total_duration:
                    audio = self.add_noise_sweep(audio, effect_pos)
                    effects_added += 1
            
            # Reverse crashes before major transitions
            if random.random() < chances['reverse_crash']:
                # Place before estimated drop/transition
                effect_pos = track_start + random.randint(24000, 48000)
                if effect_pos < total_duration - 2000:
                    audio = self.add_reverse_crash(audio, effect_pos)
                    effects_added += 1
        
        logger.info(f"Added {effects_added} DJ effects to mix")
        return audio
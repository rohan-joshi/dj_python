#!/usr/bin/env python3

import numpy as np
import random
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time

logger = logging.getLogger(__name__)

class ProfessionalProduction:
    """
    Handles professional production techniques that make mixes sound like real DJ sets
    """
    
    def __init__(self):
        self.vinyl_noise_cache = {}
        self.room_tone_cache = None
        
    def apply_professional_mastering(self, audio: AudioSegment, 
                                   mix_style: str = 'journey', 
                                   fast_mode: bool = False) -> AudioSegment:
        """Apply professional mastering chain to final mix"""
        if fast_mode:
            logger.info("Applying fast mastering (basic processing only)...")
            # Fast mode: just basic compression and normalization
            audio = compress_dynamic_range(audio, threshold=-16, ratio=2.5, attack=5, release=50)
            audio = normalize(audio, headroom=0.1)
            return audio
        
        duration_minutes = len(audio) / 1000 / 60
        logger.info(f"Applying professional mastering chain to {duration_minutes:.1f} minute mix...")
        
        # Use optimized parallel processing for large files
        if duration_minutes > 15:
            logger.info("Large mix detected - using high-performance parallel processing")
            return self._apply_parallel_mastering(audio, mix_style)
        
        # Standard processing with progress indicators
        if duration_minutes > 5:
            logger.info(f"Estimated processing time: {duration_minutes/6:.1f} minutes")
        
        start_time = time.time()
        
        # 1. Multi-band compression for glue (already parallelized)
        logger.info("Step 1/5: Multi-band compression...")
        audio = self._apply_multiband_compression(audio, mix_style)
        
        # 2. Dynamic range optimization  
        logger.info("Step 2/5: Dynamic range optimization...")
        audio = self._optimize_dynamic_range(audio, mix_style)
        
        # 3. Stereo enhancement (chunk-based for large files)
        logger.info("Step 3/5: Stereo enhancement...")
        audio = self._enhance_stereo_field(audio)
        
        # 4. Final limiting with character
        logger.info("Step 4/5: Character limiting...")
        audio = self._apply_character_limiting(audio)
        
        # 5. Normalize to professional levels
        logger.info("Step 5/5: Final normalization...")
        audio = normalize(audio, headroom=0.1)
        
        elapsed = time.time() - start_time
        logger.info(f"Professional mastering complete! ({elapsed:.1f}s)")
        return audio
    
    def _apply_parallel_mastering(self, audio: AudioSegment, mix_style: str) -> AudioSegment:
        """Apply mastering with maximum parallelization for very large files"""
        logger.info("Using high-performance parallel mastering...")
        start_time = time.time()
        
        # Step 1: Multi-band compression (already parallel)
        logger.info("Parallel Step 1/3: Multi-band compression...")
        audio = self._apply_multiband_compression(audio, mix_style)
        
        # Step 2: Process stereo enhancement and dynamic range in parallel
        # These can be done concurrently on separate copies
        logger.info("Parallel Step 2/3: Concurrent processing...")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both operations
            future_stereo = executor.submit(self._enhance_stereo_field, audio)
            future_dynamics = executor.submit(self._optimize_dynamic_range, audio, mix_style)
            
            # Get results
            stereo_enhanced = future_stereo.result()
            dynamics_optimized = future_dynamics.result()
        
        # Blend the results (average the processing)
        # Convert both to numpy and blend
        samples1 = np.array(stereo_enhanced.get_array_of_samples(), dtype=np.float32)
        samples2 = np.array(dynamics_optimized.get_array_of_samples(), dtype=np.float32)
        
        # Ensure same length
        min_len = min(len(samples1), len(samples2))
        samples1 = samples1[:min_len]
        samples2 = samples2[:min_len]
        
        # Weighted blend: 60% stereo enhancement, 40% dynamics
        blended_samples = (samples1 * 0.6 + samples2 * 0.4).astype(np.int16)
        
        blended_audio = AudioSegment(
            blended_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
        
        # Step 3: Final limiting and normalization
        logger.info("Parallel Step 3/3: Final processing...")
        blended_audio = self._apply_character_limiting(blended_audio)
        blended_audio = normalize(blended_audio, headroom=0.1)
        
        elapsed = time.time() - start_time
        speedup = (len(audio) / 1000 / 60) / elapsed * 60  # minutes of audio per minute of processing
        logger.info(f"High-performance mastering complete! ({elapsed:.1f}s, {speedup:.1f}x realtime)")
        
        return blended_audio
    
    def _apply_multiband_compression(self, audio: AudioSegment, mix_style: str) -> AudioSegment:
        """Apply multiband compression to glue the mix together - PARALLELIZED & CHUNKED"""
        start_time = time.time()
        
        # For very large files, process in chunks to avoid memory/time issues
        duration_minutes = len(audio) / 1000 / 60
        if duration_minutes > 20:  # Chunk for files > 20 minutes
            logger.info(f"Large file detected ({duration_minutes:.1f} min) - using chunked multiband compression")
            return self._apply_multiband_compression_chunked(audio, mix_style)
        
        # Split into frequency bands (this step is fast for smaller files)
        low_freq = audio.low_pass_filter(200)
        mid_freq = audio.high_pass_filter(200).low_pass_filter(3000)
        high_freq = audio.high_pass_filter(3000)
        
        # Define compression settings for each style
        compression_settings = self._get_compression_settings(mix_style)
        
        # Process each band in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all three compression jobs simultaneously
            future_low = executor.submit(
                compress_dynamic_range, low_freq, **compression_settings['low']
            )
            future_mid = executor.submit(
                compress_dynamic_range, mid_freq, **compression_settings['mid']
            )
            future_high = executor.submit(
                compress_dynamic_range, high_freq, **compression_settings['high']
            )
            
            # Collect results as they complete
            compressed_low = future_low.result()
            compressed_mid = future_mid.result()
            compressed_high = future_high.result()
        
        # Recombine with slight overlap
        result = compressed_low.overlay(compressed_mid).overlay(compressed_high)
        
        elapsed = time.time() - start_time
        logger.debug(f"Parallel multiband compression completed in {elapsed:.1f}s")
        return result
    
    def _apply_multiband_compression_chunked(self, audio: AudioSegment, mix_style: str) -> AudioSegment:
        """Apply multiband compression in chunks for very large files"""
        chunk_duration_ms = 60000  # 1-minute chunks for compression
        chunks = []
        
        # Split audio into chunks
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunks.append(chunk)
        
        logger.info(f"Processing {len(chunks)} chunks in parallel...")
        
        # Process chunks in parallel (limit workers to avoid overwhelming system)
        max_workers = min(4, mp.cpu_count())  # Limit to 4 workers max
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_chunks = list(executor.map(
                lambda chunk: self._apply_multiband_compression_single(chunk, mix_style), 
                chunks
            ))
        
        # Recombine chunks
        result = AudioSegment.empty()
        for chunk in processed_chunks:
            result += chunk
        
        return result
    
    def _apply_multiband_compression_single(self, audio: AudioSegment, mix_style: str) -> AudioSegment:
        """Apply multiband compression to a single chunk"""
        # Split into frequency bands
        low_freq = audio.low_pass_filter(200)
        mid_freq = audio.high_pass_filter(200).low_pass_filter(3000)
        high_freq = audio.high_pass_filter(3000)
        
        # Define compression settings for each style
        compression_settings = self._get_compression_settings(mix_style)
        
        # Process each band in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_low = executor.submit(
                compress_dynamic_range, low_freq, **compression_settings['low']
            )
            future_mid = executor.submit(
                compress_dynamic_range, mid_freq, **compression_settings['mid']
            )
            future_high = executor.submit(
                compress_dynamic_range, high_freq, **compression_settings['high']
            )
            
            # Get results
            compressed_low = future_low.result()
            compressed_mid = future_mid.result()
            compressed_high = future_high.result()
        
        # Combine bands
        combined_audio = compressed_low.overlay(compressed_mid).overlay(compressed_high)
        return combined_audio
    
    def _get_compression_settings(self, mix_style: str) -> Dict[str, Dict]:
        """Get compression settings for each frequency band"""
        if mix_style == 'party':
            return {
                'low': {'threshold': -16, 'ratio': 4.0, 'attack': 5, 'release': 50},
                'mid': {'threshold': -12, 'ratio': 3.0, 'attack': 3, 'release': 30},
                'high': {'threshold': -8, 'ratio': 2.5, 'attack': 1, 'release': 20}
            }
        elif mix_style == 'chill':
            return {
                'low': {'threshold': -20, 'ratio': 2.5, 'attack': 10, 'release': 100},
                'mid': {'threshold': -18, 'ratio': 2.0, 'attack': 8, 'release': 80},
                'high': {'threshold': -15, 'ratio': 1.8, 'attack': 5, 'release': 60}
            }
        else:  # journey, emotional
            return {
                'low': {'threshold': -18, 'ratio': 3.0, 'attack': 7, 'release': 70},
                'mid': {'threshold': -15, 'ratio': 2.5, 'attack': 5, 'release': 50},
                'high': {'threshold': -12, 'ratio': 2.0, 'attack': 3, 'release': 40}
            }
    
    def _optimize_dynamic_range(self, audio: AudioSegment, mix_style: str) -> AudioSegment:
        """Optimize dynamic range for the mix style"""
        if mix_style == 'party':
            # High energy: more compression, less dynamics
            return compress_dynamic_range(audio, threshold=-14, ratio=3.5, attack=3, release=40)
        elif mix_style == 'chill':
            # Preserve dynamics for relaxed feel
            return compress_dynamic_range(audio, threshold=-22, ratio=2.0, attack=15, release=120)
        elif mix_style == 'emotional':
            # Medium compression to preserve emotional peaks
            return compress_dynamic_range(audio, threshold=-18, ratio=2.8, attack=8, release=80)
        else:  # journey
            # Balanced for long listening
            return compress_dynamic_range(audio, threshold=-16, ratio=2.5, attack=6, release=60)
    
    def _enhance_stereo_field(self, audio: AudioSegment) -> AudioSegment:
        """Enhance stereo width and imaging - CHUNK-BASED for large files"""
        start_time = time.time()
        
        if audio.channels != 2:
            audio = audio.set_channels(2)
        
        # For large files, process in chunks to avoid memory issues
        duration_minutes = len(audio) / 1000 / 60
        if duration_minutes > 10:  # Process in chunks for files > 10 minutes
            return self._enhance_stereo_chunked(audio)
        
        # Standard processing for smaller files
        return self._enhance_stereo_standard(audio)
    
    def _enhance_stereo_chunked(self, audio: AudioSegment) -> AudioSegment:
        """Process stereo enhancement in parallel chunks"""
        chunk_duration_ms = 30000  # 30-second chunks
        chunks = []
        
        # Split audio into chunks
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunks.append(chunk)
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), len(chunks))) as executor:
            processed_chunks = list(executor.map(self._enhance_stereo_standard, chunks))
        
        # Recombine chunks
        result = AudioSegment.empty()
        for chunk in processed_chunks:
            result += chunk
        
        return result
    
    def _enhance_stereo_standard(self, audio: AudioSegment) -> AudioSegment:
        """Standard stereo enhancement processing"""
        # Convert to numpy for processing
        samples = audio.get_array_of_samples()
        samples = np.array(samples, dtype=np.float32)
        samples = samples.reshape((-1, 2))
        
        # Vectorized M/S processing (faster than loops)
        mid = np.mean(samples, axis=1)  # (L+R)/2
        side = np.diff(samples, axis=1).flatten()  # (L-R)
        
        # Enhance side signal slightly
        side *= 1.15
        
        # Reconstruct stereo using broadcasting
        left = mid + side
        right = mid - side
        
        # Convert back efficiently
        stereo_samples = np.column_stack((left, right))
        stereo_samples = np.clip(stereo_samples, -32768, 32767).astype(np.int16)
        stereo_samples = stereo_samples.flatten()
        
        return AudioSegment(
            stereo_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=2
        )
    
    def _apply_character_limiting(self, audio: AudioSegment) -> AudioSegment:
        """Apply final limiting with analog character"""
        # Soft limiting with gentle saturation
        return compress_dynamic_range(
            audio, 
            threshold=-3, 
            ratio=10.0,  # Heavy ratio for limiting
            attack=0.1,  # Very fast attack
            release=5    # Quick release
        )
    
    def add_vinyl_character(self, audio: AudioSegment, 
                          intensity: float = 0.3) -> AudioSegment:
        """Add subtle vinyl character (wow, flutter, surface noise)"""
        logger.debug(f"Adding vinyl character (intensity: {intensity})")
        
        # Add subtle wow and flutter
        audio = self._add_wow_flutter(audio, intensity)
        
        # Add very subtle surface noise between tracks
        audio = self._add_surface_noise(audio, intensity * 0.5)
        
        return audio
    
    def _add_wow_flutter(self, audio: AudioSegment, intensity: float) -> AudioSegment:
        """Add subtle pitch variations like vinyl wow and flutter"""
        if intensity <= 0:
            return audio
            
        samples = audio.get_array_of_samples()
        samples = np.array(samples).astype(np.float32)
        
        # Create subtle pitch modulation
        duration_seconds = len(audio) / 1000.0
        sample_rate = audio.frame_rate
        
        # Wow (slow pitch variation)
        wow_freq = 0.5 + random.uniform(-0.2, 0.2)  # ~0.3-0.7 Hz
        wow_depth = intensity * 0.002  # Very subtle
        
        # Flutter (faster pitch variation)
        flutter_freq = 8 + random.uniform(-2, 2)  # ~6-10 Hz
        flutter_depth = intensity * 0.001  # Even more subtle
        
        # Generate modulation
        t = np.linspace(0, duration_seconds, len(samples))
        wow = np.sin(2 * np.pi * wow_freq * t) * wow_depth
        flutter = np.sin(2 * np.pi * flutter_freq * t) * flutter_depth
        
        pitch_mod = 1 + wow + flutter
        
        # Apply subtle pitch modulation (simplified)
        # In practice, this would need proper pitch shifting
        # For now, we'll apply a subtle frequency-based effect
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            # Apply modulation as subtle amplitude variation
            mod_effect = 1 + (pitch_mod * 0.02).reshape(-1, 1)
            samples = samples * mod_effect
            samples = samples.flatten()
        else:
            samples = samples * (1 + pitch_mod * 0.02)
        
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(
            samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
    
    def _add_surface_noise(self, audio: AudioSegment, intensity: float) -> AudioSegment:
        """Add very subtle surface noise"""
        if intensity <= 0:
            return audio
            
        # Generate subtle pink noise
        duration_ms = len(audio)
        noise_samples = np.random.normal(0, intensity * 100, 
                                       int(audio.frame_rate * duration_ms / 1000))
        
        # Shape the noise (pink noise approximation)
        noise_samples = np.cumsum(noise_samples)
        noise_samples = noise_samples / np.max(np.abs(noise_samples)) * intensity * 500
        
        # Create noise audio segment
        noise_audio = AudioSegment(
            noise_samples.astype(np.int16).tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,
            channels=1
        )
        
        if audio.channels == 2:
            noise_audio = noise_audio.set_channels(2)
        
        # Mix very quietly with original
        return audio.overlay(noise_audio, gain_during_overlay=-40)
    
    def add_room_tone(self, audio: AudioSegment, 
                     intensity: float = 0.2) -> AudioSegment:
        """Add subtle room tone to make mix feel 'live'"""
        if intensity <= 0:
            return audio
            
        logger.debug(f"Adding room tone (intensity: {intensity})")
        
        # Generate subtle room ambience
        duration_ms = len(audio)
        room_noise = np.random.normal(0, intensity * 50, 
                                    int(audio.frame_rate * duration_ms / 1000))
        
        # Apply gentle filtering to make it sound like room tone
        # Low-pass filter effect (simplified)
        room_noise = np.convolve(room_noise, np.ones(5)/5, mode='same')
        
        room_audio = AudioSegment(
            room_noise.astype(np.int16).tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,
            channels=1
        )
        
        if audio.channels == 2:
            room_audio = room_audio.set_channels(2)
        
        # Mix very quietly
        return audio.overlay(room_audio, gain_during_overlay=-45)
    
    def create_energy_dynamics(self, audio: AudioSegment, 
                             energy_points: List[Tuple[int, float]]) -> AudioSegment:
        """Create dynamic energy variations throughout the mix"""
        logger.debug("Creating energy dynamics with volume automation")
        
        if not energy_points:
            return audio
            
        # Sort energy points by position
        energy_points = sorted(energy_points, key=lambda x: x[0])
        
        # Create volume automation curve
        duration_ms = len(audio)
        automation_curve = np.ones(duration_ms)
        
        for i, (position_ms, energy_level) in enumerate(energy_points):
            # Ensure position is within bounds
            position_ms = min(position_ms, duration_ms - 1)
            
            # Convert energy level to volume multiplier
            # energy_level: 0.0 = quiet, 1.0 = full volume, >1.0 = boosted
            volume_mult = 0.3 + (energy_level * 0.7)  # Range: 0.3 to 1.0+
            
            # Apply automation point
            automation_curve[position_ms] = volume_mult
        
        # Smooth the automation curve
        from scipy import interpolate
        positions = [ep[0] for ep in energy_points]
        volumes = [0.3 + (ep[1] * 0.7) for ep in energy_points]
        
        if len(positions) > 1:
            f = interpolate.interp1d(positions, volumes, kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')
            smooth_positions = np.linspace(0, duration_ms - 1, duration_ms)
            automation_curve = f(smooth_positions)
        
        # Apply automation to audio
        samples = audio.get_array_of_samples()
        samples = np.array(samples).astype(np.float32)
        
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            # Downsample automation curve to match audio samples
            curve_resampled = np.interp(
                np.linspace(0, len(automation_curve) - 1, len(samples)),
                np.arange(len(automation_curve)),
                automation_curve
            )
            samples = samples * curve_resampled.reshape(-1, 1)
            samples = samples.flatten()
        else:
            curve_resampled = np.interp(
                np.linspace(0, len(automation_curve) - 1, len(samples)),
                np.arange(len(automation_curve)),
                automation_curve
            )
            samples = samples * curve_resampled
        
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(
            samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
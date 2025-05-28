# 🎧 Professional DJ Mixer

An AI-powered DJ mixing system that creates professional-quality mixes using advanced transition techniques found in real SoundCloud and Spotify DJ sets.

## ✨ Features

### 🎵 **Advanced Transition Techniques**
- **Phrase-Aligned Mixing** - Transitions happen at musically correct points
- **Progressive Filter Sweeps** - Smooth frequency transitions (20Hz → 2kHz)
- **Loop Roll Transitions** - Progressive shortening (4-bar → 2-bar → 1-bar → 1/2-bar → 1/4-bar)
- **Drop-Aligned Mixing** - Perfect energy management (breakdown → buildup → drop)
- **Echo/Delay Fades** - Musical delays with reverb wash
- **Beat-Perfect Crossfades** - Equal-power crossfades aligned to beats

### 🧠 **Intelligent Audio Analysis**
- **BPM Detection** - Precise tempo analysis
- **Key Detection** - Harmonic compatibility using Camelot wheel
- **Energy Analysis** - Drop and breakdown detection
- **Beat/Phrase Detection** - Perfect cue point identification

### 🎚️ **Professional Audio Processing**
- **EQ Blending** - Bass cutting, frequency separation, bassline swapping
- **Beatmatching** - Natural tempo relationships and time-stretching
- **Professional Mastering** - Multi-band compression and stereo enhancement
- **Smart DJ Effects** - Intelligent placement of drops, air horns, and effects

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```bash
python dj_mixer.py <music_folder> --mix-style <style> --output <output.mp3>
```

### Examples

**🔥 Party Mix** (Exciting transitions, loop rolls, filter sweeps):
```bash
python dj_mixer.py lowkey_playlist/ --mix-style party --output party_mix.mp3
```

**😌 Chill Mix** (Smooth crossfades only):
```bash
python dj_mixer.py lowkey_playlist/ --mix-style chill --output chill_mix.mp3
```

**💫 Emotional Mix** (Harmonic-focused transitions):
```bash
python dj_mixer.py lowkey_playlist/ --mix-style emotional --output emotional_mix.mp3
```

**🌟 Journey Mix** (AI-selected transitions):
```bash
python dj_mixer.py lowkey_playlist/ --mix-style journey --output journey_mix.mp3
```

## 🎛️ Mix Styles

| Style | Description | Transitions Used |
|-------|-------------|------------------|
| **party** | High energy, exciting transitions | Loop rolls, filter sweeps, quick cuts |
| **chill** | Smooth and relaxed flow | Crossfades only |
| **emotional** | Harmonic and meaningful | Harmonic crossfades, echo delays |
| **journey** | AI-optimized transitions | All techniques, intelligently selected |

## 🎼 Supported Audio Formats

- MP3
- WAV  
- FLAC
- M4A
- OGG

## 🧩 System Architecture

```
dj_mixer/
├── audio_analyzer.py          # BPM, key, energy analysis
├── advanced_transitions.py    # Professional transition techniques
├── enhanced_beatmatching.py   # Harmonic mixing and tempo matching
├── professional_eq.py         # Frequency blending and EQ
├── professional_production.py # Mastering chain
├── cohesive_mix_builder.py   # Main mix creation logic
├── dj_elements.py            # DJ effects and drops
├── beat_juggling.py          # Hot cues and beat juggling
└── utils.py                  # Utility functions
```

## 🎯 Transition Selection Logic

The AI automatically selects the best transition based on:

1. **Same BPM (±5)** → Extended crossfade
2. **Key incompatible** → Filter sweep
3. **Energy increase needed** → Drop-aligned mixing  
4. **Tempo change >10 BPM** → Loop roll
5. **Mid-mix position** → Filter sweep
6. **Ambient sections** → Echo/delay fade

## 📊 What the System Does

1. **🔍 Analyzes** each track (BPM, key, energy, structure)
2. **🎼 Creates harmonic journey** (optimal track ordering)
3. **🎚️ Applies professional transitions** (phrase-aligned, beat-perfect)
4. **🎵 Adds DJ effects** (smart placement)
5. **🎧 Applies mastering** (professional sound quality)

## ⚡ Quick Test

Test with 2 tracks for faster results:
```bash
python simple_test.py
```

## 🎵 Requirements

- Python 3.8+
- librosa (audio analysis)
- pydub (audio processing) 
- numpy (numerical processing)
- essentia (advanced audio analysis)

## 📝 Notes

- **First run** may take longer due to audio analysis
- **Mastering chain** is computationally intensive (several minutes for full mixes)
- **Track ordering** is optimized for harmonic and energy flow
- **All transitions** are beat-aligned and musically correct

---

*Creates professional DJ mixes using the same techniques found in real SoundCloud and Spotify DJ sets* 🎧✨
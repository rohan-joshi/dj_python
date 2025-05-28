# Professional DJ Mixing System

An advanced Python-based DJ mixing system that creates professional-quality mixes from your music collection. This system uses sophisticated audio analysis and real DJ techniques to produce seamless, engaging mixes that rival professional DJ sets.

## Key Features

### Professional DJ Techniques
- **Advanced Beatmatching**: Precise tempo synchronization using natural tempo relationships and intelligent time-stretching
- **Harmonic Mixing**: Uses the Camelot wheel system for musically compatible track transitions
- **Phrase-Aware Mixing**: Intelligent detection of musical phrases (8, 16, 32-bar sections) for seamless transitions
- **Professional EQ Blending**: Frequency separation, bass cutting, and bassline swapping for clean mixes

### Advanced Transition Types
- **Crossfades**: Smooth volume transitions with EQ automation
- **Loop Rolls**: Creative looping sections during transitions
- **Filter Sweeps**: High/low pass filter effects for dramatic transitions  
- **Echo/Delay Fades**: Atmospheric transition effects
- **Drop-Aligned Mixing**: Perfectly timed drops and energy changes
- **Quick Cuts**: Instant transitions at perfect cue points

### Beat Juggling & Creative Mixing
- **Hot Cues**: Intelligent cue point detection for perfect entry points
- **Loop Sections**: Creative use of loops during transitions
- **Energy-Based Ordering**: Automatic track sequencing based on energy analysis
- **Dynamic Tempo Management**: Smart handling of tempo changes throughout the mix

### Technical Excellence
- **Multi-band Compression**: Professional mastering with frequency-specific compression
- **Stereo Enhancement**: Improved spatial imaging and width
- **Performance Optimization**: Parallel processing and chunked audio handling for large files
- **Professional Limiting**: Character-preserving dynamics control

## Mix Styles

### Party Mix
High-energy mixing perfect for dance floors and celebrations:
- Aggressive transitions with quick cuts and loop rolls
- Enhanced bass and punchy dynamics
- Fast-paced harmonic progressions
- Energy-driven track ordering

### Chill Mix
Relaxed, smooth mixing for background listening:
- Gentle crossfades and subtle transitions
- Warm, rounded sound character
- Gradual harmonic movements
- Flow-focused track progression

### Emotional Journey
Dynamic storytelling through music:
- Dramatic build-ups and breakdowns
- Wide dynamic range preservation
- Emotional harmonic progressions
- Narrative-driven track sequencing

### Journey Mix
Progressive exploration of musical landscapes:
- Extended transitions and creative experimentation
- Adventurous harmonic relationships
- Unexpected but musical track combinations
- Discovery-focused progression

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rohan-joshi/dj_python.git
cd dj_python

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Create a party mix from a folder of tracks
python dj_mixer.py /path/to/music/folder --mix-style party --output party_mix.mp3

# Create a chill mix
python dj_mixer.py /path/to/music/folder --mix-style chill --output chill_mix.mp3

# Create an emotional journey
python dj_mixer.py /path/to/music/folder --mix-style emotional --output journey_mix.mp3
```

## Requirements

- Python 3.8+
- librosa (audio analysis)
- pydub (audio processing)
- numpy (numerical computing)
- scipy (signal processing)

## System Architecture

The mixing system is built with modular components:

- **Audio Analyzer**: Extracts BPM, key, energy, and structural information
- **Harmonic Analyzer**: Implements Camelot wheel compatibility matching
- **Advanced Transitions**: Professional transition technique implementations
- **Professional EQ**: Musical timing-aware EQ automation
- **Beat Juggling**: Creative loop and cue point management
- **Cohesive Mix Builder**: Orchestrates the complete mixing process
- **Professional Production**: Mastering chain with performance optimizations

## Technical Details

### Audio Analysis
- BPM detection using onset strength and tempo estimation
- Key detection using chromagram analysis and template matching
- Energy analysis across frequency bands
- Structural analysis for phrase and section detection

### Harmonic Mixing
- Camelot wheel implementation for key compatibility
- Intelligent key change detection and routing
- Energy-based track ordering within harmonic groups
- Support for both major and minor key relationships

### Professional Transitions
- Beat-grid alignment for perfect synchronization
- Musical phrase detection for natural transition points
- EQ automation timed to musical structure
- Dynamic transition selection based on track characteristics

### Performance Optimizations
- Parallel processing for multi-band compression
- Chunked processing for large audio files (>20 minutes)
- Optimized numpy operations for real-time performance
- Memory-efficient audio handling

## Advanced Configuration

The system automatically detects optimal settings based on your music, but you can customize:

- Transition aggressiveness levels
- Harmonic progression preferences  
- Energy curve shaping
- Mastering chain parameters

## Performance

- **Processing Speed**: 2-3x realtime for most mixes
- **Memory Usage**: Optimized for large collections (1000+ tracks)
- **Audio Quality**: Professional mastering chain maintains broadcast standards
- **Compatibility**: Supports all common audio formats (MP3, FLAC, WAV, M4A)

## Contributing

This project welcomes contributions! Areas of interest:
- Additional transition techniques
- New mix style implementations
- Performance optimizations
- Audio analysis improvements

## License

MIT License - feel free to use this for your own projects and events!
#!/usr/bin/env python3

"""Simple test script for the enhanced DJ mixing system"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dj_mixer.audio_analyzer import AudioAnalyzer
from dj_mixer.cohesive_mix_builder import CohesiveMixBuilder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("🎵 Testing Enhanced DJ Mixing System")
    
    # Test paths
    test_files = [
        "lowkey_playlist/Anis del Mono [AKpy00uBkbg].mp3",
        "lowkey_playlist/Loukeman - Snoopy [0KUiNTPvFaU].mp3"
    ]
    
    # Initialize components
    analyzer = AudioAnalyzer()
    mix_builder = CohesiveMixBuilder()
    
    print("\n📊 Analyzing tracks...")
    analyses = {}
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"Analyzing: {file_path}")
            analysis = analyzer.analyze_track(file_path)
            analyses[file_path] = analysis
        else:
            print(f"File not found: {file_path}")
    
    if len(analyses) >= 2:
        print(f"\n🎧 Creating mix from {len(analyses)} tracks...")
        try:
            mixed_audio = mix_builder.create_beautiful_mix(analyses, 'party')
            
            # Export test mix
            output_path = "simple_test_mix.mp3"
            mixed_audio.export(output_path, format="mp3", bitrate="320k")
            print(f"✅ Mix created successfully: {output_path}")
            print(f"Duration: {len(mixed_audio)/1000:.1f} seconds")
            
        except Exception as e:
            print(f"❌ Error creating mix: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Need at least 2 tracks to create a mix")

if __name__ == "__main__":
    main()
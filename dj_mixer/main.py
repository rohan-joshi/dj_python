#!/usr/bin/env python3

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List

from .mixer import DJMixer
from .utils import AudioUtils

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='DJ Audio Mixing Script - Automatically mix audio files with tempo matching and crossfading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/music/folder
  %(prog)s track1.mp3 track2.mp3 track3.mp3 --output party_mix.wav
  %(prog)s /music --crossfade 16 --output mix.mp3 --verbose
        """
    )
    
    parser.add_argument(
        'input',
        nargs='+',
        help='Input audio files or directory containing audio files'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='dj_mix.mp3',
        help='Output file path (default: dj_mix.mp3)'
    )
    
    parser.add_argument(
        '--crossfade', '-c',
        type=int,
        default=8,
        help='Crossfade duration in beats (default: 8)'
    )
    
    parser.add_argument(
        '--tempo-tolerance', '-t',
        type=float,
        default=0.2,
        help='Maximum tempo difference as ratio (default: 0.2 = 20%%)'
    )
    
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Disable automatic track order optimization'
    )
    
    parser.add_argument(
        '--optimization-method',
        choices=['greedy', 'genetic', 'exhaustive'],
        default='greedy',
        help='Track ordering optimization method (default: greedy)'
    )
    
    parser.add_argument(
        '--professional-mode',
        action='store_true',
        default=True,
        help='Use professional DJ mixing techniques (default: enabled)'
    )
    
    parser.add_argument(
        '--basic-mode',
        action='store_true',
        help='Use basic automated mixing instead of professional DJ techniques'
    )
    
    parser.add_argument(
        '--mix-style',
        choices=['journey', 'party', 'chill', 'emotional'],
        default='journey',
        help='Overall mix style: journey (energy arc), party (high energy), chill (relaxed), emotional (harmonic focus)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['mp3', 'wav'],
        default='mp3',
        help='Output format (default: mp3)'
    )
    
    parser.add_argument(
        '--bitrate', '-b',
        default='320k',
        help='Output bitrate for MP3 (default: 320k)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Display mix statistics after completion'
    )
    
    return parser.parse_args()

def collect_audio_files(inputs: List[str]) -> List[str]:
    audio_files = []
    
    for input_path in inputs:
        if os.path.isfile(input_path):
            if AudioUtils.validate_audio_file(input_path):
                audio_files.append(os.path.abspath(input_path))
            else:
                logging.warning(f"Skipping invalid audio file: {input_path}")
        elif os.path.isdir(input_path):
            dir_files = AudioUtils.find_audio_files(input_path)
            audio_files.extend([os.path.abspath(f) for f in dir_files])
        else:
            logging.error(f"Path does not exist: {input_path}")
    
    audio_files = list(set(audio_files))
    audio_files.sort()
    
    return audio_files

def validate_output_path(output_path: str) -> str:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logging.error(f"Cannot create output directory {output_dir}: {e}")
            sys.exit(1)
    
    if os.path.exists(output_path):
        response = input(f"Output file {output_path} already exists. Overwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            logging.info("Operation cancelled by user")
            sys.exit(0)
    
    return os.path.abspath(output_path)

def print_mix_statistics(mixer: DJMixer):
    stats = mixer.get_mix_statistics()
    
    if not stats:
        logging.warning("No statistics available")
        return
    
    print("\n" + "="*50)
    print("DJ MIX STATISTICS")
    print("="*50)
    print(f"Total tracks processed: {stats['total_tracks']}")
    print(f"Average tempo: {stats['avg_tempo']:.1f} BPM")
    print(f"Tempo range: {stats['tempo_range'][0]:.1f} - {stats['tempo_range'][1]:.1f} BPM")
    print(f"Total mix duration: {stats['total_duration']:.1f} seconds ({stats['total_duration']/60:.1f} minutes)")
    print(f"Average track duration: {stats['avg_track_duration']:.1f} seconds")
    print("="*50)

def main():
    args = parse_arguments()
    
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("DJ Mixer starting...")
        
        audio_files = collect_audio_files(args.input)
        
        if not audio_files:
            logger.error("No valid audio files found")
            sys.exit(1)
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        if len(audio_files) == 1:
            logger.warning("Only one audio file found. No mixing needed.")
            audio = AudioUtils.load_audio_segment(audio_files[0])
        else:
            output_path = validate_output_path(args.output)
            
            professional_mode = args.professional_mode and not args.basic_mode
            
            mixer = DJMixer(
                crossfade_duration_beats=args.crossfade,
                tempo_tolerance=args.tempo_tolerance,
                auto_optimize=not args.no_optimize,
                professional_mode=professional_mode,
                mix_style=args.mix_style
            )
            
            if professional_mode:
                logger.info(f"Creating beautiful {args.mix_style} mix using professional techniques")
            else:
                logger.info("Using basic automated mixing")
            
            logger.info("Creating DJ mix...")
            audio = mixer.create_dj_mix(audio_files)
            
            if args.stats:
                print_mix_statistics(mixer)
        
        output_path = validate_output_path(args.output)
        
        metadata = {
            'title': f"DJ Mix - {len(audio_files)} tracks",
            'artist': 'DJ Mixer',
            'album': 'Automated DJ Mix'
        }
        
        logger.info(f"Exporting mix to {output_path}")
        AudioUtils.export_audio(
            audio, 
            output_path, 
            format=args.format,
            bitrate=args.bitrate,
            metadata=metadata
        )
        
        logger.info(f"DJ mix completed successfully: {output_path}")
        logger.info(f"Final mix duration: {len(audio) / 1000:.1f} seconds")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error creating DJ mix: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
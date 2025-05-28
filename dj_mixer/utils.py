import os
import glob
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from pydub import AudioSegment
from pydub.utils import which
import mutagen
from mutagen.id3 import ID3, TIT2, TPE1, TALB

logger = logging.getLogger(__name__)

class AudioUtils:
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg']
    
    @classmethod
    def find_audio_files(cls, path: str) -> List[str]:
        audio_files = []
        
        if os.path.isfile(path):
            if cls._is_audio_file(path):
                audio_files.append(path)
        elif os.path.isdir(path):
            for ext in cls.SUPPORTED_FORMATS:
                pattern = os.path.join(path, f"*{ext}")
                audio_files.extend(glob.glob(pattern, recursive=False))
                
                pattern = os.path.join(path, f"**/*{ext}")
                audio_files.extend(glob.glob(pattern, recursive=True))
        
        audio_files = list(set(audio_files))
        audio_files.sort()
        
        logger.info(f"Found {len(audio_files)} audio files")
        return audio_files
    
    @classmethod
    def _is_audio_file(cls, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    def load_audio_segment(cls, file_path: str) -> AudioSegment:
        try:
            audio = AudioSegment.from_file(file_path)
            logger.debug(f"Loaded {file_path}: {len(audio)}ms, {audio.frame_rate}Hz, {audio.channels} channels")
            return audio
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    @classmethod
    def normalize_audio(cls, audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
        change_in_dbfs = target_dbfs - audio.dBFS
        return audio.apply_gain(change_in_dbfs)
    
    @classmethod
    def convert_to_stereo(cls, audio: AudioSegment) -> AudioSegment:
        if audio.channels == 1:
            return audio.set_channels(2)
        return audio
    
    @classmethod
    def match_sample_rate(cls, audio: AudioSegment, target_rate: int) -> AudioSegment:
        if audio.frame_rate != target_rate:
            return audio.set_frame_rate(target_rate)
        return audio
    
    @classmethod
    def apply_fade(cls, audio: AudioSegment, fade_in_ms: int = 0, fade_out_ms: int = 0) -> AudioSegment:
        if fade_in_ms > 0:
            audio = audio.fade_in(fade_in_ms)
        if fade_out_ms > 0:
            audio = audio.fade_out(fade_out_ms)
        return audio
    
    @classmethod
    def get_audio_metadata(cls, file_path: str) -> dict:
        try:
            file = mutagen.File(file_path)
            if file is None:
                return {}
            
            metadata = {
                'title': '',
                'artist': '',
                'album': '',
                'duration': 0,
                'bitrate': 0
            }
            
            if hasattr(file, 'info') and file.info:
                metadata['duration'] = getattr(file.info, 'length', 0)
                metadata['bitrate'] = getattr(file.info, 'bitrate', 0)
            
            if 'TIT2' in file:
                metadata['title'] = str(file['TIT2'])
            elif 'TITLE' in file:
                metadata['title'] = str(file['TITLE'][0])
            
            if 'TPE1' in file:
                metadata['artist'] = str(file['TPE1'])
            elif 'ARTIST' in file:
                metadata['artist'] = str(file['ARTIST'][0])
            
            if 'TALB' in file:
                metadata['album'] = str(file['TALB'])
            elif 'ALBUM' in file:
                metadata['album'] = str(file['ALBUM'][0])
            
            return metadata
        except Exception as e:
            logger.error(f"Error reading metadata from {file_path}: {e}")
            return {}
    
    @classmethod
    def export_audio(cls, audio: AudioSegment, output_path: str, 
                    format: str = "mp3", bitrate: str = "320k",
                    metadata: Optional[dict] = None) -> None:
        try:
            export_params = {
                "format": format,
                "parameters": []
            }
            
            if format == "mp3":
                export_params["parameters"] = ["-b:a", bitrate]
            
            audio.export(output_path, **export_params)
            
            if metadata and format == "mp3":
                cls._add_metadata_to_mp3(output_path, metadata)
            
            logger.info(f"Exported audio to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting audio to {output_path}: {e}")
            raise
    
    @classmethod
    def _add_metadata_to_mp3(cls, file_path: str, metadata: dict) -> None:
        try:
            audio_file = ID3(file_path)
            
            if 'title' in metadata:
                audio_file['TIT2'] = TIT2(encoding=3, text=metadata['title'])
            if 'artist' in metadata:
                audio_file['TPE1'] = TPE1(encoding=3, text=metadata['artist'])
            if 'album' in metadata:
                audio_file['TALB'] = TALB(encoding=3, text=metadata['album'])
            
            audio_file.save()
        except Exception as e:
            logger.warning(f"Could not add metadata to {file_path}: {e}")
    
    @classmethod
    def validate_audio_file(cls, file_path: str) -> bool:
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return False
            
            if not cls._is_audio_file(file_path):
                logger.error(f"Unsupported file format: {file_path}")
                return False
            
            audio = cls.load_audio_segment(file_path)
            if len(audio) < 1000:
                logger.warning(f"Audio file is very short: {file_path}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Invalid audio file {file_path}: {e}")
            return False
    
    @classmethod
    def get_file_duration(cls, file_path: str) -> float:
        try:
            audio = cls.load_audio_segment(file_path)
            return len(audio) / 1000.0
        except Exception as e:
            logger.error(f"Error getting duration for {file_path}: {e}")
            return 0.0
    
    @classmethod
    def create_silence(cls, duration_ms: int, sample_rate: int = 44100) -> AudioSegment:
        return AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
    
    @classmethod
    def trim_audio(cls, audio: AudioSegment, start_ms: int, end_ms: int) -> AudioSegment:
        return audio[start_ms:end_ms]
import os
import time
import boto3
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import tempfile

# Load environment variables
load_dotenv()

class TranscriptionService:
    def __init__(self):
        self.region = os.getenv('AWS_REGION')
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.transcribe_client = boto3.client('transcribe', region_name=self.region)
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.intro_folder = 'intro/'
        self.meeting_folder = 'meeting/'
        self.speaker_profiles = {}
        self.silence_threshold = -40  # dB
        self.min_silence_duration = 1.0  # seconds
        
        # Create a local temp directory in the project folder
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def load_speaker_profiles(self):
        """Load speaker profiles from intro folder"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.intro_folder
            )
            
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.wav'):
                    local_path = os.path.join(self.temp_dir, os.path.basename(key))
                    self.s3_client.download_file(self.bucket_name, key, local_path)
                    
                    # Extract speaker name from filename
                    speaker_name = os.path.basename(key).split('.')[0]
                    self.speaker_profiles[speaker_name] = {
                        'audio_path': local_path,
                        'last_used': datetime.now()
                    }
        except Exception as e:
            print(f"Error loading speaker profiles: {e}")

    def is_silent_audio(self, audio_path):
        """Detect if audio is silent using RMS energy"""
        try:
            # Load audio file
            audio = AudioSegment.from_wav(audio_path)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(samples**2))
            
            # Convert to dB
            if rms == 0:
                return True
            db = 20 * np.log10(rms)
            
            return db < self.silence_threshold
        except Exception as e:
            print(f"Error detecting silence: {e}")
            return True

    def check_new_audio(self):
        """Check for new audio files in the S3 bucket"""
        try:
            # List all objects in the bucket with the meetings prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='meetings'
            )
            
            if 'Contents' not in response:
                return None
                
            # Sort files by LastModified timestamp (oldest first)
            audio_files = sorted(
                response['Contents'],
                key=lambda x: x['LastModified']
            )
            
            # Get the oldest unprocessed file
            for file in audio_files:
                if file['Key'].endswith('.wav'):
                    return file['Key']
            
            return None
            
        except Exception as e:
            print(f"Error checking for new audio: {e}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
            return None

    def transcribe_audio(self, audio_key):
        """Transcribe audio using Amazon Transcribe with speaker identification"""
        try:
            job_name = f"transcription-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            print(f"\nStarting transcription job: {job_name}")
            print(f"Audio file: {audio_key}")
            
            # Check if file exists in S3
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=audio_key)
                print("Audio file exists in S3")
            except Exception as e:
                print(f"Error: Audio file not found in S3: {e}")
                return None
            
            # Configure transcription settings
            settings = {
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 10,
                'ChannelIdentification': False,
                'ShowAlternatives': True,
                'MaxAlternatives': 3
            }
            
            print("Starting transcription with settings:", settings)
            
            # Get file extension
            file_extension = os.path.splitext(audio_key)[1].lower()
            media_format = file_extension[1:] if file_extension else 'wav'
            
            self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': f"s3://{self.bucket_name}/{audio_key}"},
                MediaFormat=media_format,
                LanguageCode='en-US',
                Settings=settings,
                OutputBucketName=self.bucket_name,
                OutputKey=f"transcripts/{job_name}.json"
            )
            
            print("Waiting for transcription to complete...")
            while True:
                try:
                    status = self.transcribe_client.get_transcription_job(
                        TranscriptionJobName=job_name
                    )
                    job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                    print(f"Current status: {job_status}")
                    
                    if job_status == 'FAILED':
                        print(f"Transcription job failed. Status: {job_status}")
                        if 'FailureReason' in status['TranscriptionJob']:
                            print(f"Failure reason: {status['TranscriptionJob']['FailureReason']}")
                        return None
                    
                    if job_status == 'COMPLETED':
                        print("Transcription job completed successfully")
                        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                        print(f"Transcript URI: {transcript_uri}")
                        
                        # Process the transcript
                        result = self.process_transcript(transcript_uri)
                        
                        # Delete the original audio file after successful transcription
                        try:
                            print(f"\nDeleting original audio file: {audio_key}")
                            self.s3_client.delete_object(
                                Bucket=self.bucket_name,
                                Key=audio_key
                            )
                            print("Audio file deleted successfully")
                        except Exception as e:
                            print(f"Error deleting audio file: {e}")
                            import traceback
                            print("Full error traceback:")
                            print(traceback.format_exc())
                        
                        return result
                    
                    time.sleep(5)
                except Exception as e:
                    print(f"Error checking transcription status: {e}")
                    import traceback
                    print("Full error traceback:")
                    print(traceback.format_exc())
                    time.sleep(5)
                    continue
                
        except Exception as e:
            print(f"Error in transcription: {e}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
            return None

    def process_transcript(self, transcript_uri):
        """Process transcript and match speakers using Amazon's speaker identification"""
        try:
            print(f"\nProcessing transcript from: {transcript_uri}")
            
            # Extract the key from the transcript URI
            transcript_key = transcript_uri.split(f"{self.bucket_name}/")[-1]
            print(f"Transcript key: {transcript_key}")
            
            # Download transcript with retries
            max_retries = 3
            retry_delay = 5
            transcript_data = None
            
            for attempt in range(max_retries):
                try:
                    # Download the transcript file from S3
                    response = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=transcript_key
                    )
                    
                    # Read the content
                    content = response['Body'].read().decode('utf-8')
                    
                    # Check if content is not empty
                    if not content:
                        print(f"Empty content received (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                        
                    transcript_data = json.loads(content)
                    print("Transcript data loaded successfully")
                    break
                except Exception as e:
                    print(f"Error downloading transcript (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise
            
            if not transcript_data:
                print("Failed to download transcript after all retries")
                return None
            
            # Check if we have the expected structure
            if 'results' not in transcript_data:
                print("Error: No 'results' in transcript data")
                print(f"Transcript data: {transcript_data}")
                return None
                
            if 'items' not in transcript_data['results']:
                print("Error: No 'items' in transcript results")
                print(f"Results: {transcript_data['results']}")
                return None
            
            # Process speaker segments
            speaker_segments = []
            current_speaker = None
            current_text = ""
            
            print("\nProcessing transcript items...")
            for item in transcript_data['results']['items']:
                if 'speaker_label' in item:
                    speaker = item['speaker_label']
                    if speaker != current_speaker:
                        if current_speaker:
                            speaker_segments.append({
                                'speaker': current_speaker,
                                'text': current_text.strip()
                            })
                        current_speaker = speaker
                        current_text = ""
                    
                if item['type'] == 'pronunciation':
                    current_text += item['alternatives'][0]['content'] + " "
            
            # Add last segment
            if current_speaker and current_text:
                speaker_segments.append({
                    'speaker': current_speaker,
                    'text': current_text.strip()
                })
            
            print(f"\nFound {len(speaker_segments)} speaker segments")
            
            # Format output
            formatted_output = []
            for segment in speaker_segments:
                # Try to match speaker with known profiles
                speaker_name = f"Speaker {segment['speaker']}"
                for profile_name in self.speaker_profiles:
                    if profile_name.lower() in segment['text'].lower():
                        speaker_name = profile_name
                        break
                
                formatted_output.append(f"{speaker_name}: {segment['text']}")
            
            if not formatted_output:
                print("Warning: No formatted output generated")
                return None
                
            result = "\n".join(formatted_output)
            print("\nGenerated transcription:")
            print(result)
            
            # Store formatted result in S3
            formatted_key = f"transcripts/{os.path.basename(transcript_key)}.txt"
            print(f"Storing formatted transcript in S3: {formatted_key}")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=formatted_key,
                Body=result,
                ContentType='text/plain'
            )
            print("Formatted transcript stored successfully in S3")
            
            return result
            
        except Exception as e:
            print(f"Error processing transcript: {e}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
            return None

    def list_all_objects(self):
        """List all objects in the bucket for debugging"""
        try:
            print("\nListing all objects in bucket:")
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in response:
                for obj in response['Contents']:
                    print(f"- {obj['Key']} (Last modified: {obj['LastModified']})")
            else:
                print("No objects found in bucket")
        except Exception as e:
            print(f"Error listing objects: {e}")

    def run(self):
        """Main loop to monitor and process audio"""
        self.load_speaker_profiles()
        
        # List all objects for debugging
        self.list_all_objects()
        
        while True:
            new_audio = self.check_new_audio()
            if new_audio:
                print(f"Processing new audio: {new_audio}")
                transcript = self.transcribe_audio(new_audio)
                if transcript:
                    print("\n=== New Transcription ===")
                    print(transcript)
                    print("=======================\n")
            time.sleep(15)

if __name__ == "__main__":
    service = TranscriptionService()
    service.run() 
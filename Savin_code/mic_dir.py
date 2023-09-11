import pyaudio
import numpy as np
import time

# 오디오 설정
CHUNK = 1024  # 한 번에 읽을 프레임 크기
FORMAT = pyaudio.paInt16  # 오디오 포맷 (16-bit)
CHANNELS = 1  # 모노 오디오
RATE = 44100  # 샘플 속도 (samples per second)

# PyAudio 객체 생성
p = pyaudio.PyAudio()

# 입력 스트림 열기 (첫 번째 마이크)
stream1 = p.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=CHUNK)

# 입력 스트림 열기 (두 번째 마이크)
stream2 = p.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=CHUNK)

try:
    while True:
        # 오디오 데이터 읽기
        data1 = stream1.read(CHUNK)
        data2 = stream2.read(CHUNK)
        
        # 데이터를 NumPy 배열로 변환
        audio_data1 = np.frombuffer(data1, dtype=np.int16)
        audio_data2 = np.frombuffer(data2, dtype=np.int16)
        
        # 각 마이크의 음량 계산
        volume1 = np.abs(audio_data1).mean()
        volume2 = np.abs(audio_data2).mean()
        
        # 각 마이크의 볼륨을 백분율로 계산
        total_volume = volume1 + volume2
        if total_volume > 0:
            ratio1 = (volume1 / total_volume) * 100
            ratio2 = (volume2 / total_volume) * 100
        else:
            ratio1 = 0
            ratio2 = 0
        
        # 백분율로 된 볼륨 출력
        print(f"마이크 1: {ratio1:.2f}%, 마이크 2: {ratio2:.2f}%")
        
        # 3초 대기
        time.sleep(3)
        
except KeyboardInterrupt:
    pass

print("오디오 스트리밍 종료.")

# 스트림과 PyAudio 객체 닫기
stream1.stop_stream()
stream1.close()
stream2.stop_stream()
stream2.close()
p.terminate()
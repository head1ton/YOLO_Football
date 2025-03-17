import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        # 카메라 움직임을 감지하기 위한 취소 거리 설정
        self.minimum_distance = 5

        # Lucas-Kanade method parameters
        self.lk_params = dict(
            winSize=(15, 15), # 윈도우 크기
            maxLevel=2, # 피라미드 레벨
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # 첫번째 프레임을 그레이스 스케일로 변환
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 특징점을 추적할 영역을 정의하는 마스크 생성 (경기장 위와 아래)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        # 특징점 추적을 위한 파라미터 설정
        self.features = dict(
            maxCorners=100, # 최대 코너 수
            qualityLevel=0.3,   # 코너 품질 수준
            minDistance=3,  # 최소 거리
            blockSize=7,    # 블록 크기
            mask=mask_features  # 마스크
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # 카메라 움직임을 저장할 리스트 초기화
        camera_movement = [[0,0]] * len(frames)

        # 첫번째 프레임을 그레이 스케일로 변환
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # 첫번째 프레임에서 특징점 추적
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # 각 프레임에 대해 반복
        for frame_num in range(1, len(frames)):
            # 현재 프레임을 그레이 스케일로 변환
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            # 현재 프레임에서 Lucas-Kanade method로 특징점 추적
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # 각 특징점 쌍에 대해 거리 계산
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            # 최대 거리가 최소 거리보다 크면 카메라 움직임 기록
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # 현재 프레임을 이전 프레임으로 설정
            old_gray = frame_gray.copy()

        # stub 경로로 camera_movement pickling
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f'Camera Movement X: {x_movement:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f'Camera Movement Y: {y_movement:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames
import cv2
import sys
sys.path.append('../')
from utils import measure_distance, get_foot_position


class SpeedAndDistance_Estimator():
    def __init__(self):
        # 프레임 윈도우 크기 설정 (5 프레임마다 계산)
        self.frame_window = 5
        # 프레임 속도 설정 (초당 24프레임)
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        # 각 객체의 총 이동 거리를 저장할 딕셔너리 초기화
        total_distance = {}

        # 각 객체에 대해 반복
        for object, object_tracks in tracks.items():
            # 공 또는 심판인 경우 제외
            if object == "ball" or object == "referees":
                continue

            # 객체의 총 프레임 수
            number_of_frames = len(object_tracks)
            # 프레임 윈도우 단위로 반복
            for frame_num in range(0, number_of_frames, self.frame_window):
                # 마지막 프레임 번호 계산
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                # 현재 프레임의 각 트랙 ID에 대해 반복
                for track_id, _ in object_tracks[frame_num].items():
                    # 마지막 프레임에 트랙 ID가 없는 경우 제외
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # 시작 위치와 끝 위치 (ViewTransformer add_transformed_position_to_tracks에서 저장된 position_transformed)
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # 위치 정보가 없으면 건너띔
                    if start_position is None or end_position is None:
                        continue

                    # 이동 거리 계산
                    distance_covered = measure_distance(start_position, end_position)
                    # 경과 시간 계산
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    # 속도 계산 (m/s)
                    speed_meters_per_second = distance_covered / time_elapsed
                    # 속도 계산 (km/h) 3.6은 1시간을 초로 환산하는 상수
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # 객체의 총 이동 거리 초기화
                    if object not in total_distance:
                        total_distance[object] = {}

                    # 트랙 ID의 총 이동 거리 초기화
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    # 총 이동 거리에 현재 이동 거리 추가
                    total_distance[object][track_id] += distance_covered

                    # 프레임 윈도우 내의 모든 프레임에 대해 속도 및 이동 거리 추가
                    for frame_num_batch in range(frame_num, last_frame):
                        # 현재 프레임에 트랙 ID가 없는 경우 제외
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        # 속도와 이동 거리를 트랙 정보에 추가
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_spped_and_distance(self, frames, tracks):
        # 출력 프레임 리스트 초기화
        output_frames = []
        # 각 프레임에 대해 반복
        for frame_num, frame in enumerate(frames):
            # 각 객체에 대해 반복
            for object, object_tracks in tracks.items():
                # 공과 심판은 제외
                if object == "ball" or object == "referees":
                    continue
                # 현재 프레임의 트랙 정보에 대해 반복
                for _, track_info in object_tracks[frame_num].items():
                    # 속도 정보가 있다면
                    if "speed" in track_info:
                        # 속도와 이동 거리
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue
                        # 바운딩 박스
                        bbox = track_info['bbox']
                        # 발 위치 계산
                        position = get_foot_position(bbox)
                        # 위치를 리스트로 변환
                        position = list(position)
                        # 텍스트 위치 조정
                        position[1] += 40

                        # 위치를 정수형 튜플로 변환
                        position = tuple(map(int, position))
                        # 속도 텍스트 추가
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        # 이동 거리 텍스트 추가
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            # 출력 프레임 리스트에 현재 프레임 추가
            output_frames.append(frame)
        # 출력 프레임 리스트 반환
        return output_frames
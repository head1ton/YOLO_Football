import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        # 코트의 너비와 길이를 설정
        court_width = 68
        court_length = 23.32

        # 픽셀 좌표로 정의된 코트의 꼭짓점들
        self.pixel_vertices = np.array([[100, 1035],
                                        [265, 275],
                                        [910, 260],
                                        [1640, 915]])

        # 실제 코드의 꼭지점들 (타겟 좌표)
        self.target_vertices = np.array([
            [0, court_width],   # 왼쪽 아래
            [0, 0],             # 왼쪽 위
            [court_length, 0],  # 오른쪽 위
            [court_length, court_width] # 오른쪽 아래
        ])

        # 꼭지점 좌표를 float32로 변환
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # 픽셀 좌표에서 타겟 좌표로의 투시 변환 행렬 계산
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        # 포인트를 정수형으로 변환하여 다각형 내부에 있는지 확인
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # 포인트를 변환하기 위해 reshape하고 float32 타입으로 변환
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        # 투시 변환을 적용하여 포인트 변환
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        # 변환된 포인트를 2D(원래 형태)로 reshape하여 반환
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        # 각 객체(선수, 공, 심판)에 대해 반복
        for object, object_tracks in tracks.items():
            # 각 프레임에 대해 반복
            for frame_num, track in enumerate(object_tracks):
                # 각 트랙 ID에 대해 반복
                for track_id, track_info in track.items():
                    # 조정된 위치를 가져옴 (카메라 add_adjust_positions_to_tracks 에서 저장된 position_adjusted)
                    position = track_info['position_adjusted']
                    # 위치를 numpy 배열로 변환
                    position = np.array(position)
                    # 위치를 변환
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        # 변환된 위치를 리스트로 변환하여 저장
                        position_transformed = position_transformed.squeeze().tolist()
                    # 변환된 위치를 트랙 데이터에 저장
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
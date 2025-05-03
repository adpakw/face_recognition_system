CREATE TABLE IF NOT EXISTS video_analysis_results (
    video_name TEXT NOT NULL,
    frame_number INT NOT NULL,
    timestamp FLOAT NOT NULL,

    person_bbox_x1 INT,
    person_bbox_x2 INT,
    person_bbox_y1 INT,
    person_bbox_y2 INT,
    person_detection_conf FLOAT,
    
    face_bbox_x1 INT,
    face_bbox_x2 INT,
    face_bbox_y1 INT,
    face_bbox_y2 INT,
    face_detection_conf FLOAT,
    
    person_name TEXT,
    person_identification_conf FLOAT
);
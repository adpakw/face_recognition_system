CREATE TABLE IF NOT EXISTS videos (
    video_name TEXT NOT NULL,
    original_fps FLOAT NOT NULL,
    width INT NOT NULL,
    height INT NOT NULL,
    total_frames INT NOT NULL,
    duration FLOAT NOT NULL,
    processing_date TIMESTAMP NOT NULL,
    status_of_processing TEXT NOT NULL
);
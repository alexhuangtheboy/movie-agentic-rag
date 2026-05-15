CREATE TABLE IF NOT EXISTS title_content (
    movie_id text PRIMARY KEY,
    titleType text,
    primaryTitle text,
    originalTitle text,
    isAdult boolean,
    startYear integer,
    endYear integer,
    runtimeMinutes integer,
    genres text
);

CREATE TABLE IF NOT EXISTS movie_directors (
    movie_id text REFERENCES title_content(movie_id),
    nconst text,
    primaryName text,
    birthYear integer,
    deathYear integer,
    primaryProfession text
);

CREATE INDEX IF NOT EXISTS idx_movie_directors_movie_id ON movie_directors(movie_id);
CREATE INDEX IF NOT EXISTS idx_movie_directors_name ON movie_directors(primaryName);

CREATE TABLE IF NOT EXISTS movie_writers (
    movie_id text REFERENCES title_content(movie_id),
    nconst text,
    primaryName text,
    birthYear integer,
    deathYear integer,
    primaryProfession text
);

CREATE INDEX IF NOT EXISTS idx_movie_writers_movie_id ON movie_writers(movie_id);
CREATE INDEX IF NOT EXISTS idx_movie_writers_name ON movie_writers(primaryName);

CREATE TABLE IF NOT EXISTS movie_ratings (
    movie_id text REFERENCES title_content(movie_id),
    averageRating numeric,
    numVotes integer
);

CREATE INDEX IF NOT EXISTS idx_movie_ratings_movie_id ON movie_ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_movie_ratings_average ON movie_ratings(averageRating);
CREATE INDEX IF NOT EXISTS idx_title_content_genres ON title_content(genres);
CREATE INDEX IF NOT EXISTS idx_title_content_year ON title_content(startYear);

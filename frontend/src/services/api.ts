const API_BASE_URL = "http://127.0.0.1:8000";

export interface UploadResponse {
  video_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  duration: number;
}

export interface VideoRecord {
  video_id: string;
  filename: string;
  status: string;
  created_at: string;
  chunk_count?: number;
  duration?: number;
  error?: string;
}

export interface Evidence {
  start: number;
  end: number;
  text: string;
  relevance_score: number;
  chunk_count: number;
}

export interface SearchResponse {
  video_id: string;
  question: string;
  answer: string;
  evidence: Evidence[];
}


export async function uploadVideo(
  file: File
): Promise<UploadResponse> {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/videos`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const error = await response.json();

    throw new Error(
      error.detail || "Failed to upload video."
    );
  }

  return response.json();
}


export async function getVideos(): Promise<VideoRecord[]> {
  const response = await fetch(
    `${API_BASE_URL}/videos`
  );

  if (!response.ok) {
    const error = await response.json();

    throw new Error(
      error.detail || "Failed to load video history."
    );
  }

  return response.json();
}


export function getVideoUrl(
  videoId: string
): string {
  return `${API_BASE_URL}/videos/${videoId}/file`;
}


export async function searchVideo(
  videoId: string,
  question: string
): Promise<SearchResponse> {
  const response = await fetch(
    `${API_BASE_URL}/videos/${videoId}/search`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
      }),
    }
  );

  if (!response.ok) {
    const error = await response.json();

    throw new Error(
      error.detail || "Failed to search video."
    );
  }

  return response.json();
}
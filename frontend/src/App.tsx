import { useRef, useState } from "react";

import {
  Upload,
  Search,
  Video,
  Sparkles,
  Clock,
  Loader2,
  Play,
} from "lucide-react";

import "./App.css";

import {
  uploadVideo,
  searchVideo,
  getVideos,
  getVideoUrl,
  type Evidence,
  type VideoRecord,
} from "./services/api";


function formatTimestamp(
  seconds: number
): string {
  const totalSeconds =
    Math.floor(seconds);

  const hours =
    Math.floor(
      totalSeconds / 3600
    );

  const minutes =
    Math.floor(
      (totalSeconds % 3600) / 60
    );

  const remainingSeconds =
    totalSeconds % 60;

  if (hours > 0) {
    return (
      `${String(hours).padStart(2, "0")}:` +
      `${String(minutes).padStart(2, "0")}:` +
      `${String(remainingSeconds).padStart(2, "0")}`
    );
  }

  return (
    `${String(minutes).padStart(2, "0")}:` +
    `${String(remainingSeconds).padStart(2, "0")}`
  );
}


function App() {
  const fileInputRef =
    useRef<HTMLInputElement>(null);

  const videoPlayerRef =
    useRef<HTMLVideoElement>(null);


  const [question, setQuestion] =
    useState("");

  const [uploading, setUploading] =
    useState(false);

  const [searching, setSearching] =
    useState(false);

  const [videoId, setVideoId] =
    useState<string | null>(null);

  const [videoName, setVideoName] =
    useState<string | null>(null);

  const [videoUrl, setVideoUrl] =
    useState<string | null>(null);

  const [answer, setAnswer] =
    useState<string | null>(null);

  const [evidence, setEvidence] =
    useState<Evidence[]>([]);

  const [error, setError] =
    useState<string | null>(null);


  const [activeView, setActiveView] =
    useState<"videos" | "history">(
      "videos"
    );

  const [history, setHistory] =
    useState<VideoRecord[]>([]);

  const [loadingHistory, setLoadingHistory] =
    useState(false);


  function openFilePicker() {
    fileInputRef.current?.click();
  }


  async function handleFileSelected(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    setError(null);
    setAnswer(null);
    setEvidence([]);
    setUploading(true);

    try {
      const result =
        await uploadVideo(file);

      setVideoId(
        result.video_id
      );

      setVideoName(
        result.filename
      );

      const localVideoUrl =
        URL.createObjectURL(file);

      setVideoUrl(
        localVideoUrl
      );

      /*
       * Refresh history after a
       * successful upload.
       */
      const videos =
        await getVideos();

      setHistory(videos);

      setActiveView("videos");

    } catch (uploadError) {
      console.error(uploadError);

      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Failed to upload video."
      );

    } finally {
      setUploading(false);
    }
  }


  async function handleSearch() {
    if (
      !videoId ||
      !question.trim() ||
      searching
    ) {
      return;
    }

    setError(null);
    setAnswer(null);
    setEvidence([]);
    setSearching(true);

    try {
      const result =
        await searchVideo(
          videoId,
          question.trim()
        );

      setAnswer(
        result.answer
      );

      setEvidence(
        result.evidence
      );

    } catch (searchError) {
      console.error(searchError);

      setError(
        searchError instanceof Error
          ? searchError.message
          : "Failed to search video."
      );

    } finally {
      setSearching(false);
    }
  }


  async function handleHistoryClick() {
    setActiveView("history");
    setError(null);
    setLoadingHistory(true);

    try {
      const videos =
        await getVideos();

      setHistory(videos);

    } catch (historyError) {
      console.error(historyError);

      setError(
        historyError instanceof Error
          ? historyError.message
          : "Failed to load video history."
      );

    } finally {
      setLoadingHistory(false);
    }
  }


  function handleVideosClick() {
    setActiveView("videos");
    setError(null);
  }


  function openHistoryVideo(
    video: VideoRecord
  ) {
    if (video.status !== "ready") {
      return;
    }

    setActiveView("videos");

    setVideoId(
      video.video_id
    );

    setVideoName(
      video.filename
    );

    setVideoUrl(
      getVideoUrl(video.video_id)
    );

    setAnswer(null);
    setEvidence([]);
    setQuestion("");
    setError(null);
  }


  function handleQuestionKeyDown(
    event: React.KeyboardEvent<HTMLInputElement>
  ) {
    if (
      event.key === "Enter" &&
      question.trim()
    ) {
      handleSearch();
    }
  }


  function jumpToTimestamp(
    timestamp: number
  ) {
    if (!videoPlayerRef.current) {
      return;
    }

    videoPlayerRef.current.currentTime =
      timestamp;

    videoPlayerRef.current.play();
  }


  return (
    <div className="app">

      {/* Hidden file input */}

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={
          handleFileSelected
        }
        style={{
          display: "none",
        }}
      />


      {/* Sidebar */}

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">
            <Sparkles size={20} />
          </div>

          <span className="brand-name">
            <span className="brand-clip">
              Clip
            </span>
            <span className="brand-wise">
              wise
            </span>
          </span>

        </div>


        <nav className="sidebar-nav">

          <button
            className={
              `nav-item ${
                activeView === "videos"
                  ? "active"
                  : ""
              }`
            }
            onClick={
              handleVideosClick
            }
          >
            <Video size={18} />
            Videos
          </button>


          <button
            className={
              `nav-item ${
                activeView === "history"
                  ? "active"
                  : ""
              }`
            }
            onClick={
              handleHistoryClick
            }
          >
            <Clock size={18} />
            History
          </button>

        </nav>

      </aside>


      {/* Main */}

      <main className="main">

        <header className="topbar">

          <div>

            <h1>
              {activeView === "history"
                ? "Video History"
                : "Video Intelligence"}
            </h1>

            <p>
              {activeView === "history"
                ? "View your previously uploaded videos."
                : "Understand and search your videos using AI."}
            </p>

          </div>


          <button
            className="upload-button"
            onClick={
              openFilePicker
            }
            disabled={
              uploading
            }
          >

            {uploading ? (
              <>
                <Loader2
                  size={18}
                  className="spin"
                />

                Processing...
              </>
            ) : (
              <>
                <Upload size={18} />

                Upload Video
              </>
            )}

          </button>

        </header>


        <section className="workspace">

          {activeView === "history" ? (

            /* =========================
               HISTORY VIEW
               ========================= */

            <div className="history-page">

              <div className="history-header">

                <div>
                  <h2>
                    Video History
                  </h2>

                  <p>
                    Previously uploaded videos
                  </p>
                </div>

              </div>


              {loadingHistory ? (

                <div className="history-empty">

                  <Loader2
                    size={32}
                    className="spin"
                  />

                  <h3>
                    Loading history...
                  </h3>

                </div>

              ) : history.length === 0 ? (

                <div className="history-empty">

                  <Clock size={36} />

                  <h3>
                    No videos yet
                  </h3>

                  <p>
                    Upload a video to see it here.
                  </p>

                </div>

              ) : (

                <div className="history-list">

                  {history.map(
                    (video) => (

                      <button
                        key={
                          video.video_id
                        }
                        className="history-card"
                        onClick={() =>
                          openHistoryVideo(
                            video
                          )
                        }
                        disabled={
                          video.status !==
                          "ready"
                        }
                      >

                        <div className="history-card-icon">

                          <Video
                            size={22}
                          />

                        </div>


                        <div className="history-card-info">

                          <h3>
                            {video.filename}
                          </h3>

                          <p>
                            {new Date(
                              video.created_at
                            ).toLocaleString()}
                          </p>

                        </div>


                        <div
                          className={
                            `history-status ${
                              video.status
                            }`
                          }
                        >
                          {video.status}
                        </div>

                      </button>

                    )
                  )}

                </div>

              )}

            </div>

          ) : (

            /* =========================
               VIDEO + AI VIEW
               ========================= */

            <>

              {/* Video section */}

              <div className="video-section">

                {videoUrl ? (

                  <div className="video-container">

                    <video
                      ref={
                        videoPlayerRef
                      }
                      className="video-player"
                      src={videoUrl}
                      controls
                    />


                    <div className="video-info">

                      <div>

                        <h2>
                          {videoName}
                        </h2>

                        <p>
                          Video ID:{" "}
                          {videoId}
                        </p>

                      </div>


                      <span className="status-badge">
                        Ready
                      </span>

                    </div>

                  </div>

                ) : (

                  <div className="video-placeholder">

                    <Video size={48} />

                    <h2>
                      No video selected
                    </h2>

                    <p>
                      Upload a video to start
                      analyzing it.
                    </p>


                    <button
                      className="upload-large"
                      onClick={
                        openFilePicker
                      }
                      disabled={
                        uploading
                      }
                    >

                      {uploading ? (
                        <>
                          <Loader2
                            size={18}
                            className="spin"
                          />

                          Processing...
                        </>
                      ) : (
                        <>
                          <Upload
                            size={18}
                          />

                          Upload Video
                        </>
                      )}

                    </button>


                    {error && (
                      <div className="error-message">
                        {error}
                      </div>
                    )}

                  </div>

                )}

              </div>


              {/* AI section */}

              <div className="ai-section">

                <div className="ai-header">

                  <div>

                    <h2>
                      <Sparkles
                        size={18}
                      />

                      Ask Clipwise
                    </h2>

                    <p>
                      Ask anything about your video.
                    </p>

                  </div>

                </div>


                {/* Chat / answer area */}

                <div className="chat-area">

                  {!answer &&
                  !searching ? (

                    <div className="empty-chat">

                      <Search
                        size={32}
                      />

                      <h3>
                        Ask a question
                      </h3>

                      <p>
                        Clipwise will search the
                        video, find the relevant
                        moments, and generate
                        an answer.
                      </p>

                    </div>

                  ) : searching ? (

                    <div className="empty-chat">

                      <Loader2
                        size={32}
                        className="spin"
                      />

                      <h3>
                        Searching video...
                      </h3>

                      <p>
                        Clipwise is finding the
                        most relevant moments.
                      </p>

                    </div>

                  ) : (

                    <div className="answer-area">

                      <div className="answer-label">

                        <Sparkles
                          size={15}
                        />

                        Clipwise Answer

                      </div>


                      <div className="answer-text">
                        {answer}
                      </div>


                      {evidence.length > 0 && (

                        <div className="evidence-section">

                          <div className="evidence-title">
                            Supporting moments
                          </div>


                          <div className="evidence-list">

                            {evidence.map(
                              (
                                item,
                                index
                              ) => (

                                <button
                                  key={
                                    `${item.start}-${index}`
                                  }
                                  className="evidence-card"
                                  onClick={() =>
                                    jumpToTimestamp(
                                      item.start
                                    )
                                  }
                                >

                                  <div className="evidence-top">

                                    <span>

                                      <Play
                                        size={12}
                                        fill="currentColor"
                                      />

                                      {formatTimestamp(
                                        item.start
                                      )}

                                      {" - "}

                                      {formatTimestamp(
                                        item.end
                                      )}

                                    </span>

                                  </div>


                                  <p>
                                    {item.text}
                                  </p>

                                </button>

                              )
                            )}

                          </div>

                        </div>

                      )}

                    </div>

                  )}

                </div>


                {/* Error */}

                {error &&
                  videoId && (

                    <div className="search-error">
                      {error}
                    </div>

                  )}


                {/* Question box */}

                <div className="question-box">

                  <input
                    type="text"
                    placeholder={
                      videoId
                        ? "Ask something about this video..."
                        : "Upload a video first..."
                    }
                    value={
                      question
                    }
                    onChange={(
                      event
                    ) =>
                      setQuestion(
                        event.target.value
                      )
                    }
                    onKeyDown={
                      handleQuestionKeyDown
                    }
                    disabled={
                      !videoId ||
                      searching
                    }
                  />


                  <button
                    onClick={
                      handleSearch
                    }
                    disabled={
                      !question.trim() ||
                      !videoId ||
                      searching
                    }
                  >

                    {searching ? (
                      <Loader2
                        size={18}
                        className="spin"
                      />
                    ) : (
                      <Search
                        size={18}
                      />
                    )}

                  </button>

                </div>

              </div>

            </>

          )}

        </section>

      </main>

    </div>
  );
}


export default App;
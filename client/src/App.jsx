// import React, { useState, useEffect, useRef } from 'react';
// import { FaPaperPlane, FaBars, FaTrashAlt, FaVolumeUp, FaSyncAlt, FaPlus } from 'react-icons/fa';
// import hljs from 'highlight.js';
// import 'highlight.js/styles/a11y-dark.css';
// import { Loader } from "@react-three/drei";
// import { Canvas } from "@react-three/fiber";
// import { Leva } from "leva";
// import { Experience } from "./components/Expereince";
// // import './App.css'

// const App = () => {
//   const [messages, setMessages] = useState([]);
//   const [inputMessage, setInputMessage] = useState('');
//   const [sessions, setSessions] = useState([]);
//   const [currentSessionId, setCurrentSessionId] = useState(null);
//   const [currentEndpoint, setCurrentEndpoint] = useState('http://127.0.0.1:8000/ask-sk');
//   const [isSidebarOpen, setIsSidebarOpen] = useState(false);
//   const textareaRef = useRef(null);

//   useEffect(() => {
//     hljs.configure({ cssSelector: 'pre code' });
//     hljs.highlightAll();
//     loadChatSessions();
//   }, []);

//   useEffect(() => {
//     if (textareaRef.current) {
//       textareaRef.current.style.height = 'auto';
//       textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
//     }
//   }, [inputMessage]);

//   const loadChatSessions = async () => {
//     try {
//       const response = await fetch("http://127.0.0.1:8000/sessions");
//       if (!response.ok) throw new Error(await response.text());
//       const sessions = await response.json();
//       setSessions(sessions);
//     } catch (error) {
//       console.error("Failed to load sessions:", error);
//       addMessage(`Error loading history: ${error.message}`, false);
//     }
//   };

//   const loadSessionMessages = async (sessionId) => {
//     try {
//       const response = await fetch(`http://127.0.0.1:8000/sessions/${sessionId}`);
//       if (!response.ok) throw new Error(await response.text());
//       const session = await response.json();
//       setCurrentSessionId(sessionId);
//       setMessages(session.messages.map(msg => ({
//         ...msg,
//         timestamp: new Date(msg.timestamp)
//       })));
//       setIsSidebarOpen(false);
//     } catch (error) {
//       console.error("Failed to load session:", error);
//       addMessage(`Error loading session: ${error.message}`, false);
//     }
//   };

//   const deleteSession = async (sessionId) => {
//     try {
//       const response = await fetch(`http://127.0.0.1:8000/sessions/${sessionId}`, { method: 'DELETE' });
//       if (!response.ok) throw new Error(await response.text());
//       await loadChatSessions();
//       if (currentSessionId === sessionId) {
//         setCurrentSessionId(null);
//         setMessages([]);
//       }
//     } catch (error) {
//       console.error("Failed to delete session:", error);
//       addMessage(`Error deleting session: ${error.message}`, false);
//     }
//   };

//   const addMessage = (content, is_user, timestamp = new Date(), audio = null) => {
//     const formattedContent = content.replace(
//       /```(\w+)?\s*([\s\S]+?)```/g,
//       (match, lang, code) => {
//         const language = lang || 'plaintext';
//         return `<pre><code class="language-${language}">${code.trim()}</code></pre>`;
//       }
//     );

//     setMessages(prev => [...prev, {
//       content: formattedContent,
//       is_user,
//       timestamp,
//       audio,
//       rawContent: content
//     }]);
//   };

//   const playAudio = (audioData) => {
//     const audioSrc = `data:audio/mpeg;base64,${audioData}`;
//     const audio = new Audio(audioSrc);
//     audio.play();
//   };

//   const handleSendMessage = async () => {
//     const message = inputMessage.trim();
//     if (!message) return;

//     try {
//       addMessage(message, true);
//       setInputMessage('');

//       const formData = new FormData();
//       formData.append('question', message);

//       const response = await fetch(currentEndpoint, {
//         method: "POST",
//         headers: { 'Session-ID': currentSessionId || '' },
//         body: formData
//       });

//       if (!response.ok) throw new Error(await response.text());
//       const data = await response.json();

//       if (data.session_id) {
//         setCurrentSessionId(data.session_id);
//         loadChatSessions();
//       }

//       // Add the message with audio
//       addMessage(data.answer, false, new Date(), data.audio);

//     } catch (error) {
//       addMessage(`Error: ${error.message}`, false);
//     }
//   };

//   const lastAudioTimestamp = useRef(null);
//   const isMounted = useRef(false);
//   useEffect(() => {
//     isMounted.current = true;
//     return () => { isMounted.current = false; };
//   }, []);

//   useEffect(() => {
//     if (!isMounted.current) return;

//     // Find the most recent bot message with audio
//     const botMessages = messages.filter(msg => !msg.is_user);
//     const lastBotMessage = botMessages[botMessages.length - 1];

//     if (lastBotMessage?.audio && lastBotMessage.timestamp > lastAudioTimestamp.current) {
//       playAudio(lastBotMessage.audio);
//       lastAudioTimestamp.current = lastBotMessage.timestamp;
//     }
//   }, [messages]);
//   return (
//     <>
//       <Canvas shadows camera={{ position: [0, 0, 1], fov: 30 }}>
//         <Experience />
//       </Canvas>
//     </>
//     // <div className="h-100" >
//     // <div className="container-fluid vh-100 g-100">
//     //   <div className="row g-100 h-100">
//     //     {/* History Sidebar */}
//     //     <div className={`col-md-3 col-lg-2 history-sidebar p-0 ${isSidebarOpen ? 'active' : ''}`}>
//     //       <div className="history-header p-3 d-flex justify-content-between align-items-center">
//     //         <h5 className="text-white mb-0">Chat History</h5>
//     //       </div>
//     //       <div className="history-actions p-2 border-bottom">
//     //         {/* <button className="btn btn-sm btn-outline-light w-100" onClick={loadChatSessions}> */}
//     //         <button className="btn btn-sm btn-outline-light w-100" onClick={() => window.location.reload()}>
//     //           <FaSyncAlt className="mr-2" /> Refresh
//     //         </button>
//     //       </div>
//     //       <div className="history-list" id="historyPanel">
//     //         {sessions.length === 0 ? (
//     //           <div className="p-3 text-muted">No chat history found</div>
//     //         ) : (
//     //           sessions.map(sessionId => (
//     //             <div key={sessionId} className="history-item p-3 border-bottom">
//     //               <div className="d-flex justify-content-between align-items-center">
//     //                 <div
//     //                   className="text-truncate session-link"
//     //                   style={{ color: '#ececf1', cursor: 'pointer' }}
//     //                   onClick={() => loadSessionMessages(sessionId)}
//     //                 >
//     //                   {sessionId.substring(0, 12)}...
//     //                 </div>
//     //                 <button
//     //                   className="btn btn-link text-danger p-0"
//     //                   onClick={() => deleteSession(sessionId)}
//     //                 >
//     //                   <FaTrashAlt />
//     //                 </button>
//     //               </div>
//     //             </div>
//     //           ))
//     //         )}
//     //       </div>
//     //     </div>

//     //     {/* Main Chat Area */}
//     //     <div className="col-md-9 col-lg-10 h-100" style={{ backgroundColor: '#343541' }}>
//     //       <div className="d-flex flex-column h-100">
//     //         <div className="chat-header p-3 d-flex align-items-center">
//     //           <button
//     //             className="btn btn-link text-white mr-2 d-md-none"
//     //             onClick={() => setIsSidebarOpen(!isSidebarOpen)}
//     //           >
//     //             <FaBars />
//     //           </button>
//     //           <h3 className="text-white mb-0 mr-auto">Ebla ChatBot Assistant</h3>
//     //           <select
//     //             className="form-control endpoint-select"
//     //             value={currentEndpoint}
//     //             onChange={(e) => setCurrentEndpoint(e.target.value)}
//     //             style={{ width: '120px' }}
//     //           >
//     //             <option value="http://127.0.0.1:8000/ask-rag">RAG</option>
//     //             <option value="http://127.0.0.1:8000/ask-sk">SK</option>
//     //           </select>
//     //         </div>

//     //         <div className="chat-box flex-grow-1 overflow-auto p-3">
//     //           {messages.map((msg, index) => (
//     //             <div key={index} className={`message ${msg.is_user ? 'user' : 'bot'}`}>
//     //               <div className="message-content">
//     //                 <div className="message-icon">
//     //                   <img src={`/static/images/${msg.is_user ? 'user' : 'gpt'}.png`} alt="avatar" />
//     //                 </div>
//     //                 <div className="message-body">
//     //                   <div className="message-text" dangerouslySetInnerHTML={{ __html: msg.content }} />
//     //                   <div className="message-footer">
//     //                     <div className="message-time">
//     //                       {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
//     //                     </div>
//     //                     {/* {!msg.is_user && msg.audio && (
//     //                       <button className="play-audio-btn" onClick={() => playAudio(msg.audio)}>
//     //                         <FaVolumeUp />
//     //                       </button>
//     //                     )} */}
//     //                   </div>
//     //                 </div>
//     //               </div>
//     //             </div>
//     //           ))}
//     //         </div>

//     //         <div className="chat-input p-3">
//     //           <div className="input-group">
//     //             <textarea
//     //               ref={textareaRef}
//     //               className="form-control"
//     //               rows="1"
//     //               placeholder="Type your message..."
//     //               value={inputMessage}
//     //               onChange={(e) => setInputMessage(e.target.value)}
//     //               onKeyDown={(e) => {
//     //                 if (e.key === 'Enter' && !e.shiftKey) {
//     //                   e.preventDefault();
//     //                   handleSendMessage();
//     //                 }
//     //               }}
//     //             />
//     //             <div className="input-group-append">
//     //               <button className="btn btn-primary" onClick={handleSendMessage}>
//     //                 <FaPaperPlane />
//     //               </button>
//     //             </div>
//     //           </div>
//     //         </div>
//     //       </div>
//     //     </div>
//     //   </div>
//     // </div>
//     // </div>
//   );
// };

// export default App;


import { Loader } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Leva } from "leva";
import { Experience } from "./components/Experience";
import { UI } from "./components/UI";

function App() {
  return (
    <>
      <Loader />
      <Leva hidden />
      <UI />
      <Canvas shadows camera={{ position: [0, 0, 1], fov: 30 }}>
        <Experience />
      </Canvas>
    </>
  );
}

export default App;
// import { useRef } from "react";
// import { useChat } from "../hooks/useChat";

// export const UI = ({ hidden, ...props }) => {
//     const input = useRef();
//     const { chat, loading, cameraZoomed, setCameraZoomed, message } = useChat();

//     const sendMessage = () => {
//         const text = input.current.value;
//         if (!loading && !message) {
//             chat(text);
//             input.current.value = "";
//         }
//     };
//     if (hidden) {
//         return null;
//     }

//     return (
//         <>
//             <div className="fixed top-0 left-0 right-0 bottom-0 z-10 flex justify-between p-4 flex-col pointer-events-none">
//                 <div className="self-start backdrop-blur-md bg-white bg-opacity-50 p-4 rounded-lg">
//                     <h1 align="center" className="font-black text-xl">Ebla Chat-Bot Assistant</h1>
//                     <p align="center">Here to help you 🫡</p><br />
//                     <h1 align="center" className="font-black text-xl">Special Commands</h1>
//                     <p align="center">Dance</p>
//                 </div>
//                 <div className="w-full flex flex-col items-end justify-center gap-4">
//                     <button
//                         onClick={() => setCameraZoomed(!cameraZoomed)}
//                         className="pointer-events-auto bg-blue-500 hover:bg-blue-600 text-white p-4 rounded-md"
//                     >
//                         {cameraZoomed ? (
//                             <svg
//                                 xmlns="http://www.w3.org/2000/svg"
//                                 fill="none"
//                                 viewBox="0 0 24 24"
//                                 strokeWidth={1.5}
//                                 stroke="currentColor"
//                                 className="w-6 h-6"
//                             >
//                                 <path
//                                     strokeLinecap="round"
//                                     strokeLinejoin="round"
//                                     d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM13.5 10.5h-6"
//                                 />
//                             </svg>
//                         ) : (
//                             <svg
//                                 xmlns="http://www.w3.org/2000/svg"
//                                 fill="none"
//                                 viewBox="0 0 24 24"
//                                 strokeWidth={1.5}
//                                 stroke="currentColor"
//                                 className="w-6 h-6"
//                             >
//                                 <path
//                                     strokeLinecap="round"
//                                     strokeLinejoin="round"
//                                     d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM10.5 7.5v6m3-3h-6"
//                                 />
//                             </svg>
//                         )}
//                     </button>
//                     {/* <button
//                         onClick={stopAnimation}
//                         className="pointer-events-auto bg-blue-500 hover:bg-blue-600 text-white p-4 rounded-md"
//                     >
//                         <svg
//                             xmlns="http://www.w3.org/2000/svg"
//                             fill="none"
//                             viewBox="0 0 24 24"
//                             strokeWidth={1.5}
//                             stroke="currentColor"
//                             className="w-6 h-6"
//                         >
//                             <path
//                                 strokeLinecap="round"
//                                 d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
//                             />
//                         </svg>
//                     </button> */}
//                 </div>
//                 <div className="flex items-center gap-2 pointer-events-auto max-w-screen-sm w-full mx-auto">
//                     <input
//                         className="w-full placeholder:text-gray-800 placeholder:italic p-4 rounded-md bg-opacity-50 bg-white backdrop-blur-md"
//                         placeholder="Type a message..."
//                         ref={input}
//                         onKeyDown={(e) => {
//                             if (e.key === "Enter") {
//                                 sendMessage();
//                             }
//                         }}
//                     />
//                     <button
//                         disabled={loading || message}
//                         onClick={sendMessage}
//                         className={`bg-blue-500 hover:bg-blue-600 text-white p-4 px-10 font-semibold uppercase rounded-md ${loading || message ? "cursor-not-allowed opacity-30" : ""
//                             }`}
//                     >
//                         Send
//                     </button>
//                 </div>
//             </div>
//         </>
//     );
// };

import { useRef, useState, useEffect } from "react";
import { useChat } from "../hooks/useChat";

export const UI = ({ hidden, ...props }) => {
    const input = useRef();
    const { chat, loading, cameraZoomed, setCameraZoomed, message, setSession } = useChat();
    const [sessions, setSessions] = useState([]);
    const [selectedSession, setSelectedSession] = useState(null);

    useEffect(() => {
        // Fetch chat sessions when component mounts
        const fetchSessions = async () => {
            try {
                const res = await fetch("http://localhost:8000/sessions");
                const data = await res.json();
                setSessions(data);
            } catch (error) {
                console.error('Error fetching sessions:', error);
            }
        };
        fetchSessions();
    }, [chat]);

    const sendMessage = () => {
        const text = input.current.value;
        if (!loading && !message) {
            chat(text);
            input.current.value = "";
        }
    };

    if (hidden) {
        return null;
    }

    return (
        <>
            {/* Chat History Sidebar */}
            <div className="fixed top-0 left-0 bottom-0 w-64 bg-white bg-opacity-50 backdrop-blur-md p-1 z-9 overflow-y-auto cursor-pointer">
                <h1 className="font-black text-xl" align='center'>Chat History</h1>

                <div className="flex flex-col space-y-2">
                    {sessions.map((sessionId) => (
                        <button
                            key={sessionId}
                            onClick={() => {
                                setSelectedSession(sessionId);
                                setSession(sessionId);
                            }}
                            className={`w-full text-left px-4 py-2 rounded-lg transition 
          ${sessionId === selectedSession
                                    ? "bg-blue-500 text-white"
                                    : "bg-gray-100 text-gray-800 hover:bg-blue-100 hover:text-blue-800"}
        focus:outline-none focus:ring-2 focus:ring-blue-400`}
                        >
                            {sessionId}
                        </button>
                    ))}
                </div>
            </div>

            {/* Original UI shifted right */}
            <div className="fixed top-0 left-64 right-0 bottom-0 z-10 flex justify-between p-4 flex-col pointer-events-none">
                {/* Rest of the original UI remains unchanged */}
                <div className="self-start backdrop-blur-md bg-white bg-opacity-50 p-4 rounded-lg">
                    <h1 align="center" className="font-black text-xl">Ebla Chat-Bot Assistant</h1>
                    <p align="center">Here to help you 🫡</p><br />
                    <h1 align="center" className="font-black text-xl">Special Commands</h1>
                    <p align="center">Dance</p>
                </div>
                <div className="w-full flex flex-col items-end justify-center gap-4">
                    <button
                        onClick={() => setCameraZoomed(!cameraZoomed)}
                        className="pointer-events-auto bg-blue-500 hover:bg-blue-600 text-white p-4 rounded-md"
                    >
                        {cameraZoomed ? (
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                                strokeWidth={1.5}
                                stroke="currentColor"
                                className="w-6 h-6"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM13.5 10.5h-6"
                                />
                            </svg>
                        ) : (
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                                strokeWidth={1.5}
                                stroke="currentColor"
                                className="w-6 h-6"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM10.5 7.5v6m3-3h-6"
                                />
                            </svg>
                        )}
                    </button>
                </div>
                <div className="flex items-center gap-2 pointer-events-auto max-w-screen-sm w-full mx-auto">
                    <input
                        className="w-full placeholder:text-gray-800 placeholder:italic p-4 rounded-md bg-opacity-50 bg-white backdrop-blur-md"
                        placeholder="Type a message..."
                        ref={input}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") {
                                sendMessage();
                            }
                        }}
                    />
                    <button
                        disabled={loading || message}
                        onClick={sendMessage}
                        className={`bg-blue-500 hover:bg-blue-600 text-white p-4 px-10 font-semibold uppercase rounded-md ${loading || message ? "cursor-not-allowed opacity-30" : ""
                            }`}
                    >
                        Send
                    </button>
                </div>
            </div>
        </>
    );
};
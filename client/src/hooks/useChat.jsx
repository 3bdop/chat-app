// import { createContext, useContext, useEffect, useState } from "react";

// const backendUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";

// const ChatContext = createContext();

// export const ChatProvider = ({ children }) => {
//     const chat = async (message) => {

//         setLoading(true);
//         const formData = new FormData();
//         formData.append("question", message);
//         const response = await fetch(`${backendUrl}/ask-sk`, {
//             method: "POST",
//             body: formData
//         });
//         console.log(response)
//         // const resp = (await data.json()).messages;
//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }

//         const data = await response.json();

//         // setMessages((messages) => [...messages, ...data]);
//         const newMessage = {
//             content: data.answer,
//             audio: data.audio,
//             animation: "talking",
//             facialExpression: "smile",
//         };

//         setMessages(prev => [...prev, newMessage]);
//         setLoading(false);
//     };
//     const [messages, setMessages] = useState([]);
//     // const [message, setMessage] = useState();

//     const [currentMessage, setCurrentMessage] = useState(null);
//     const [loading, setLoading] = useState(false);
//     const [cameraZoomed, setCameraZoomed] = useState(true);

//     const onMessagePlayed = () => {
//         // setMessages((messages) => messages.slice(1));
//         setMessages(prev => prev.slice(1));
//     };

//     useEffect(() => {
//         setCurrentMessage(messages[0] || null);

//     }, [messages]);

//     return (
//         <ChatContext.Provider
//             value={{
//                 // chat,
//                 // message,
//                 // onMessagePlayed,
//                 // loading,
//                 // cameraZoomed,
//                 // setCameraZoomed,
//                 chat,
//                 message: currentMessage,
//                 onMessagePlayed,
//                 loading,
//                 cameraZoomed,
//                 setCameraZoomed,
//                 messages
//             }}
//         >
//             {children}
//         </ChatContext.Provider>
//     );
// };

// export const useChat = () => {
//     const context = useContext(ChatContext);
//     if (!context) {
//         throw new Error("useChat must be used within a ChatProvider");
//     }
//     return context;
// };

import { createContext, useContext, useEffect, useState } from "react";

const backendUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";

const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
    const [messages, setMessages] = useState([]);
    const [currentMessage, setCurrentMessage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [cameraZoomed, setCameraZoomed] = useState(true);

    const chat = async (message) => {
        setLoading(true);
        const formData = new FormData();
        formData.append("question", message);

        try {
            const response = await fetch(`${backendUrl}/ask-sk`, {
                method: "POST",
                body: formData,
                headers: {
                    "Session-ID": localStorage.getItem("sessionId") || ""
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Store session ID if it's a new session
            if (!localStorage.getItem("sessionId")) {
                localStorage.setItem("sessionId", data.session_id);
            }

            // Transform backend messages to frontend format
            const newMessages = data.messages.map(msg => ({
                content: msg.text_en,
                audio: msg.audio,
                lipsync: msg.lipsync,
                animation: msg.animation,
                facialExpression: msg.facialExpression
            }));

            setMessages(prev => [...prev, ...newMessages]);

        } catch (error) {
            console.error("Chat error:", error);
        } finally {
            setLoading(false);
        }
    };

    const onMessagePlayed = () => {
        setMessages(prev => prev.slice(1));
    };

    useEffect(() => {
        setCurrentMessage(messages[0] || null);
    }, [messages]);

    return (
        <ChatContext.Provider
            value={{
                chat,
                message: currentMessage,
                onMessagePlayed,
                loading,
                cameraZoomed,
                setCameraZoomed,
                messages
            }}
        >
            {children}
        </ChatContext.Provider>
    );
};

export const useChat = () => {
    const context = useContext(ChatContext);
    if (!context) {
        throw new Error("useChat must be used within a ChatProvider");
    }
    return context;
};
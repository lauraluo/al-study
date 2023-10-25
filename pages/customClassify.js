import { useCallback, useEffect, useLayoutEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import * as MobileNet from "@tensorflow-models/mobilenet";
import Head from "next/head";

const w = 400;
const h = 400;

export async function getServerSideProps(context) {
    return {
        props: {
            data: 1,
        },
    };
}

export default function DemoCustomClassify({ data }) {
    console.log(data);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const netRef = useRef(null);

    useLayoutEffect(() => {
        const runObjectDetection = async () => {
            const video = videoRef.current;

            video.play();

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {},
            });
            video.srcObject = stream;

            const net = await MobileNet.load();

            netRef.current = net;
        };

        runObjectDetection();

        return () => {
            const video = videoRef.current;
            if (video) {
                video.srcObject.getTracks().forEach((track) => track.stop());
            }
        };
    }, []);

    const handlerVideoOnloadEnd = useCallback(() => {
        if (!videoRef.current || !canvasRef.current || !netRef.current) {
            return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const net = netRef.current;

        canvas.width = video.width || w;
        canvas.height = video.height || h;

        const detectFrame = async () => {
            const predictions = await net.classify(video);
            const ctx = canvas.getContext("2d");

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            predictions.forEach((prediction, index) => {
                ctx.beginPath();
                ctx.fillStyle = "red";
                ctx.font = "18px Arial";
                ctx.fillText(
                    `${prediction.className} (${Math.round(
                        prediction.probability * 100,
                    )}%)`,
                    10,
                    20 * index + 20,
                );
                ctx.lineWidth = 2;
                ctx.strokeStyle = "red";
                ctx.fillStyle = "red";
                ctx.stroke();
            });

            requestAnimationFrame(detectFrame);
        };

        detectFrame();
    }, [videoRef, canvasRef]);

    return (
        <div className="root">
            <video
                ref={videoRef}
                autoPlay
                onTimeUpdate={handlerVideoOnloadEnd}
            />
            <canvas ref={canvasRef} />

            <style jsx>{``}</style>
        </div>
    );
}

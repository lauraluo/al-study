import { useEffect, useLayoutEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as MobileNet from "@tensorflow-models/mobilenet";
import Head from "next/head";

import styles from "../styles/Home.module.css";

export default function Home() {
    const videoElRef = useRef(null);
    const [predictionsData, setPredictionsData] = useState([]);
    const [active, setActive] = useState(true);

    useLayoutEffect(() => {
        // 加载 MobileNet 模型
        let mobilenet;
        async function loadMobilenet() {
            mobilenet = await MobileNet.load();
        }

        // 获取摄像头视频流
        async function startWebcam() {
            const video = videoElRef.current;

            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
            });
            video.srcObject = stream;

            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    resolve(video);
                };
            });
        }

        // 对摄像头捕获的帧进行实时分类
        async function classifyFrame(video) {
            const predictions = await mobilenet.classify(video);

            setPredictionsData(predictions);
        }

        // 开始实时分类
        async function startClassification() {
            await loadMobilenet();
            const video = await startWebcam();

            video.addEventListener("loadeddata", () => {
                setInterval(() => {
                    if (active) {
                        classifyFrame(video);
                    }
                }, 1000); // 每秒进行一次分类
            });
        }

        startClassification();
    }, []);

    return (
        <div className={styles.container}>
            <Head>
                <title>Create Next App</title>
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <main>
                <video autoPlay playsInline ref={videoElRef} />
            </main>

            <ul className="predictionItems">
                {predictionsData.map((prediction, index) => (
                    <li key={index}>
                        <dt>{prediction.className}</dt>
                        <dd>{prediction.probability}</dd>
                    </li>
                ))}
            </ul>

            {/* <div className="ctrls">
                <button>Start</button>
                <button>Stop</button>
            </div> */}

            <style jsx>{`
                main {
                    padding: 5rem 0;
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }

                .predictionItems li {
                    list-style: none;
                    padding: 0;
                    display: flex;
                }

                .predictionItems li dt {
                    margin-right: 1rem;
                    font-weight: bold;
                }

                .predictionItems li dd {
                    text-align: right;
                }
            `}</style>

            <style jsx global>{`
                html,
                body {
                    padding: 0;
                    margin: 0;
                    font-family: -apple-system, BlinkMacSystemFont, Segoe UI,
                        Roboto, Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans,
                        Helvetica Neue, sans-serif;
                }
            `}</style>
        </div>
    );
}

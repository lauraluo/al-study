import {
    useCallback,
    useEffect,
    useLayoutEffect,
    useRef,
    useState,
} from "react";
import * as tf from "@tensorflow/tfjs";
import * as MobileNet from "@tensorflow-models/mobilenet";
import Head from "next/head";

export async function getServerSideProps(context) {
    return {
        props: {
            data: "",
        },
    };
}

export default function DemoCustomClassify({ data }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const netRef = useRef(null);

    const datasetRef = useRef({});
    const webcamStreamRef = useRef(null);
    const [tfReady, setTfReady] = useState(false);
    const [isVideoReady, setVideoReady] = useState(false);

    useEffect(() => {
        const enableModuleAndWebCam = async () => {
            const video = videoRef.current;

            try {
                const webcamStream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                });

                video.srcObject = webcamStream;

                await video.play();

                const net = await MobileNet.load();

                netRef.current = net;

                setTfReady(true);
            } catch (error) {
                console.error("無法啟用網路攝像頭:", error);
            }
        };

        enableModuleAndWebCam();

        return () => {};
    }, [videoRef.current]);

    const getNormalizedLogitsFromWebcam = () => {
        return tf.tidy(() => {
            // Make a prediction through mobilenet and flatten to a vector.
            const result = netRef.current
                .infer(tf.browser.fromPixels(videoRef.current), "conv_preds")
                .flatten();

            // Normalize the result to unit length.
            return result.div(result.norm());
        });
    };

    const addExample = (classId) => {
        /**
         * Reads a frame from the webcam, feeds it through MobileNet, normalizes it
         */
        tf.tidy(() => {
            // Compute the logits vector from the current webcam frame.
            const logits = getNormalizedLogitsFromWebcam();
            // Reshape the logits so its a [1, 1000] matrix instead of a [1000] vector. This allows us to concatenate it
            // to the dataset.
            const logits2d = tf.expandDims(logits, 0);

            const dataset = datasetRef.current;

            if (dataset[classId] == null) {
                // No matrix has been defined for the class yet, store a [1, 1000] matrix.
                dataset[classId] = tf.keep(logits2d);
            } else {
                // Concatenate the logits with the matrix for the class, creating an [N_prev + 1, 1000] matrix.
                dataset[classId] = tf.keep(
                    dataset[classId].concat(logits2d, 0),
                );
            }

            Object.keys(dataset).forEach((key) => {
                console.log(key);
                console.log(dataset[key].arraySync());
            });
        });
    };

    const predict = () => {
        if (!isVideoReady || !tfReady || !Object.keys(datasetRef.current).length) {
            return;
        }

        // const datasetCollectionTensor = tf.tensor(Object.keys(datasetRef.current).map((key) => {
        //     return datasetRef.current[key].arraySync();
        // }));
        const dataset = datasetRef.current;

        let datasetCollectionTensor = null;

        Object.keys(dataset).forEach((key) => {
            if (!datasetCollectionTensor) {
                datasetCollectionTensor = dataset[key].expandDims(0);
            } else {
                datasetCollectionTensor.concat(dataset[key].expandDims(0), 0);
            }
        });

        console.log(datasetCollectionTensor.arraySync());
        // const similarities = tf.tidy(() => {
        //     // Get the logits from the webcam and reshape it to a matrix of [1, 1000].
        //     const logits = getNormalizedLogitsFromWebcam().expandDims(1);
        //     // Compte the matrix multiply of the dataset and the logits to compute similarities.
        //     // This is a vector of shape [N].

        //     return datasetConcatenated.matMul(logits);
        // });
    };

    const isReady = tfReady && isVideoReady;

    return (
        <div>
            <h2>影像分類</h2>
            <hr />
            <button
                disabled={!isReady}
                onClick={() => {
                    addExample("HAPPY");
                }}
            >
                加入 HAPPY 的樣本
            </button>
            <button
                disabled={!isReady}
                onClick={() => {
                    addExample("SAD");
                }}
            >
                加入 SAD 的樣本
            </button>
            <button disabled={!isReady} onClick={() => {
                predict();
            }}>預測</button>
            <hr />
            <div>
                {isReady ? <div>截入完成</div> : <div>載入中...</div>}
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    onLoadStart={() => {
                        setVideoReady(true);
                    }}
                />
            </div>
        </div>
    );
}

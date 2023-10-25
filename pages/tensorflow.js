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
    const [kValue, setK] = useState(3);
    const totalExamplesClassRef = useRef([]);
    const datasetRef = useRef({});
    const webcamStreamRef = useRef(null);
    const [tfReady, setTfReady] = useState(false);
    const [result, setResult] = useState(null);
    const [isVideoReady, setVideoReady] = useState(false);

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
        });
        totalExamplesClassRef.current.push(classId);
    };

    const topk = (values, k) => {
        /**
         * Given an unsorted array of values, compute the top k indices and values.
         */
        const valuesAndIndices = [];
        for (let i = 0; i < values.length; i++) {
            valuesAndIndices.push({ value: values[i], index: i });
        }
        valuesAndIndices.sort((a, b) => {
            return b.value - a.value;
        });
        const topkValues = new Float32Array(k);
        const topkIndices = new Int32Array(k);
        for (let i = 0; i < k; i++) {
            topkValues[i] = valuesAndIndices[i].value;
            topkIndices[i] = valuesAndIndices[i].index;
        }
        return { values: topkValues, indices: topkIndices };
    };

    const predict = async () => {
        if (
            !isVideoReady ||
            !tfReady ||
            !Object.keys(datasetRef.current).length
        ) {
            return;
        }

        const similarities = tf.tidy(() => {
            const dataset = datasetRef.current;

            let datasetCollectionTensor = null;

            Object.keys(dataset).forEach((key) => {
                if (!datasetCollectionTensor) {
                    datasetCollectionTensor = tf.keep(dataset[key]);
                } else {
                    datasetCollectionTensor = tf.keep(
                        datasetCollectionTensor.concat(dataset[key], 0),
                    );
                }
            });

            // Get the logits from the webcam and reshape it to a matrix of [1, 1000].
            const logits = getNormalizedLogitsFromWebcam().expandDims(1);
            // Compte the matrix multiply of the dataset and the logits to compute similarities.
            // This is a vector of shape [N].
            return datasetCollectionTensor.matMul(logits);
        });

        const values = await similarities.data();
        const expSize = totalExamplesClassRef.current.length;
        // Compute the top k indices and values in our similarities vector.
        const top = topk(values, Math.min(expSize, kValue));
        // Compute the winner.
        const topKClassSet = [];
        const topKClassSum = {};

        top.indices.forEach((expIndex) => {
            topKClassSet.push(totalExamplesClassRef.current[expIndex]);
        });

        topKClassSet.forEach((classId) => {
            topKClassSum[classId] = topKClassSum[classId]
                ? topKClassSet[classId] + 1
                : 1;
        });

        let topClass = null;
        let maxFrequency = 0;

        Object.keys(topKClassSum).forEach((classId) => {
            const frequency = topKClassSum[classId];

            if (frequency > maxFrequency) {
                topClass = classId;
                maxFrequency = frequency;
            }
        });

        console.log(topClass);

        similarities.dispose();

        setResult(topClass);
    };

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

    useEffect(() => {
        if (!tfReady) {
            return;
        }

        const interval = setInterval(() => {
            predict();
        }, 1000);

        return () => {
            clearInterval(interval);
        };
    }, [tfReady]);

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
            <button
                disabled={!isReady}
                onClick={() => {
                    predict();
                }}
            >
                預測
            </button>
            <hr />
            <div>
                {isReady ? <div>截入完成</div> : <div>載入中...</div>}
                <div>
                    K 值：
                    <span>{kValue}</span>{" "}
                    <input
                        type="range"
                        min="1"
                        max="11"
                        value={kValue}
                        step={1}
                        onChange={(e) => {
                            setK(e.target.value);
                        }}
                    />
                </div>
                {result && <div style={{ background: 'green', color: 'white'}}>預測結果是： {result}</div>}
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

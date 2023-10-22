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

import styles from "../styles/Home.module.css";

export default function Home() {
    const videoElRef = useRef(null);
    const [predictionsData, setPredictionsData] = useState([]);
    const [active, setActive] = useState(false);
    const calculateTopClass = (topKIndices, kVal) => {
        /**
         * Computes the most likely class from a topk calculation.
         */

        let imageClass = -1;
        const confidences = {};

        if (topKIndices == null) {
            // No class predicted
            return { classIndex: imageClass, confidences };
        }

        const indicesForClasses = [];
        const topKCountsForClasses = [];
        for (const i in dataset) {
            topKCountsForClasses.push(0);
            let num = classExampleCount[i];
            if (+i > 0) {
                num += indicesForClasses[+i - 1];
            }
            indicesForClasses.push(num);
        }

        for (let i = 0; i < topKIndices.length; i++) {
            for (
                let classForEntry = 0;
                classForEntry < indicesForClasses.length;
                classForEntry++
            ) {
                if (topKIndices[i] < indicesForClasses[classForEntry]) {
                    topKCountsForClasses[classForEntry]++;
                    break;
                }
            }
        }

        let topConfidence = 0;
        for (const i in dataset) {
            const probability = topKCountsForClasses[i] / kVal;
            if (probability > topConfidence) {
                topConfidence = probability;
                imageClass = +i;
            }
            confidences[i] = probability;
        }

        return { classIndex: imageClass, confidences };
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

    const addExample = (classId) => {
        /**
         * Reads a frame from the webcam, feeds it through MobileNet, normalizes it
         */
        tf.tidy(() => {
            // Compute the logits vector from the current webcam frame.
            const logits = getNormalizedLogitsFromWebcam();
            // Reshape the logits so its a [1, 1000] matrix instead of a [1000] vector. This allows us to concatenate it
            // to the dataset.
            const logits2d = logits.expandDims(0);

            if (dataset[classId] == null) {
                // No matrix has been defined for the class yet, store a [1, 1000] matrix.
                dataset[classId] = tf.keep(logits2d);
            } else {
                // Concatenate the logits with the matrix for the class, creating an [N_prev + 1, 1000] matrix.
                const oldDataset = dataset[classId];
                dataset[classId] = tf.keep(
                    dataset[classId].concat(logits2d, 0),
                );
            }

            // Observable hack to update the dataset.
            const res = dataset;
            // mutable dataset = res;
        });

        const predict = async () => {
            const similarities = tf.tidy(() => {
                // Get the logits from the webcam and reshape it to a matrix of [1, 1000].
                const logits = getNormalizedLogitsFromWebcam().expandDims(1);
                // Compte the matrix multiply of the dataset and the logits to compute similarities.
                // This is a vector of shape [N].
                return datasetConcatenated.matMul(logits);
            });

            const values = await similarities.data();
            // Compute the top k indices and values in our similarities vector.
            const top = topk(values, Math.min(totalExamples, K_demo));
            // Compute the winner.
            const topClass = calculateTopClass(top.indices, K_demo);
            similarities.dispose();
            return topClass;
        };
    };

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

    const getNormalizedLogitsFromWebcam = () => {
        return tf.tidy(async () => {
            // Make a prediction through mobilenet and flatten to a vector.
            const webcam = await startWebcam();
            const result = mobilenet
                .infer(tf.fromPixels(webcam), "conv_preds")
                .flatten();

            // Normalize the result to unit length.
            return result.div(result.norm());
        });
    };

    return (
        <div className={styles.container}>
            <Head>
                <title>Create Next App</title>
                <link rel="icon" href="/favicon.ico" />
            </Head>

            {/* <main>
                <video autoPlay playsInline ref={videoElRef} />
            </main>

            <ul className="predictionItems">
                {predictionsData.map((prediction, index) => (
                    <li key={index}>
                        <dt>{prediction.className}</dt>
                        <dd>{prediction.probability}</dd>
                    </li>
                ))}
            </ul> */}

            <div className="ctrls">
                <button>Start</button>
                <button>Stop</button>
            </div>

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

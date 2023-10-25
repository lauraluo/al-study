import { useEffect, useLayoutEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as MobileNet from "@tensorflow-models/mobilenet";
import Head from "next/head";

import styles from "../styles/Home.module.css";

export default function Home() {
    const videoElRef = useRef(null);
    const [predictionsData, setPredictionsData] = useState([]);
    const [active, setActive] = useState(true);



    return (
        <div className={styles.container}>
            <Head>
                <title>Create Next App</title>
                <link rel="icon" href="/favicon.ico" />
            </Head>




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

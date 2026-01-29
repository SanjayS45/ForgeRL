import { useState, useCallback, useRef } from 'react';
import axios from 'axios';

export function useRewardModelTraining(apiBase) {
  const [isTraining, setIsTraining] = useState(false);
  const [history, setHistory] = useState({ loss: [], accuracy: [] });
  const pollRef = useRef(null);

  const startTraining = useCallback(
    async (datasetName, params, onComplete) => {
      setIsTraining(true);
      setHistory({ loss: [], accuracy: [] });

      try {
        const res = await axios.post(
          `${apiBase}/reward-model/train?dataset_name=${datasetName}`,
          params
        );
        const jobId = res.data.job_id;

        pollRef.current = setInterval(async () => {
          try {
            const statusRes = await axios.get(
              `${apiBase}/training/${jobId}`
            );
            const status = statusRes.data;

            if (status.train_loss !== null) {
              setHistory((prev) => ({
                loss: [
                  ...prev.loss,
                  { epoch: prev.loss.length + 1, value: status.train_loss },
                ],
                accuracy: [
                  ...prev.accuracy,
                  {
                    epoch: prev.accuracy.length + 1,
                    value: (status.train_accuracy || 0) * 100,
                  },
                ],
              }));
            }

            if (status.status === 'completed' || status.status === 'failed') {
              clearInterval(pollRef.current);
              setIsTraining(false);
              if (onComplete) onComplete(status);
            }
          } catch (e) {
            console.error('Polling error:', e);
          }
        }, 1500);
      } catch (error) {
        setIsTraining(false);
        throw error;
      }
    },
    [apiBase]
  );

  const stopTraining = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
    }
    setIsTraining(false);
  }, []);

  return { isTraining, history, startTraining, stopTraining };
}

export function usePolicyTraining(apiBase) {
  const [isTraining, setIsTraining] = useState(false);
  const [history, setHistory] = useState({ reward: [], loss: [] });
  const pollRef = useRef(null);

  const startTraining = useCallback(
    async (params, onComplete) => {
      setIsTraining(true);
      setHistory({ reward: [], loss: [] });

      try {
        const res = await axios.post(`${apiBase}/policy/train`, params);
        const jobId = res.data.job_id;

        pollRef.current = setInterval(async () => {
          try {
            const statusRes = await axios.get(
              `${apiBase}/training/${jobId}`
            );
            const status = statusRes.data;

            if (status.cumulative_reward !== null) {
              setHistory((prev) => ({
                reward: [
                  ...prev.reward,
                  { step: prev.reward.length + 1, value: status.cumulative_reward },
                ],
                loss: [
                  ...prev.loss,
                  { step: prev.loss.length + 1, value: status.policy_loss || 0 },
                ],
              }));
            }

            if (status.status === 'completed' || status.status === 'failed') {
              clearInterval(pollRef.current);
              setIsTraining(false);
              if (onComplete) onComplete(status);
            }
          } catch (e) {
            console.error('Polling error:', e);
          }
        }, 2000);
      } catch (error) {
        setIsTraining(false);
        throw error;
      }
    },
    [apiBase]
  );

  const stopTraining = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
    }
    setIsTraining(false);
  }, []);

  return { isTraining, history, startTraining, stopTraining };
}

export default { useRewardModelTraining, usePolicyTraining };


import { useCallback, useRef } from 'react';
import { useGraphStore } from '../store/Usegraphstore';
import { createStepSession } from '../api/api';

export function useStepSession() {
  const sessionRef = useRef(null);
  const {
    setRunMode, setStepPhase, setCurrentStep,
    setNodeStatus, setNodeMetrics, setAllNodeStatuses,
    setPendingBranches, setCompletedBranches,
    setStepSession, runConfig, inputFile,
    nodes, edges,
  } = useGraphStore();

  const handleMessage = useCallback((msg) => {
    switch (msg.event) {
      case 'ready':
        setRunMode('stepping');
        setStepPhase('forward');
        setCurrentStep(msg.layer_id);
        setNodeStatus(msg.layer_id, 'pending');
        break;

      case 'step_done':
        setNodeStatus(msg.layer_id, 'done');
        if (msg.metrics) setNodeMetrics(msg.layer_id, msg.metrics);
        setCurrentStep(msg.next_layer_id ?? null);
        if (msg.next_layer_id) setNodeStatus(msg.next_layer_id, 'pending');
        break;

      case 'branch_point':
        setCurrentStep(msg.layer_id);
        setPendingBranches(msg.branches);
        setCompletedBranches([]);
        setNodeStatus(msg.layer_id, 'branch');
        break;

      case 'branch_done':
        setCompletedBranches(prev => [...prev, msg.branch]);
        break;

      case 'branches_complete':
        setPendingBranches([]);
        setNodeStatus(msg.layer_id, 'done');
        break;

      case 'forward_complete':
        setStepPhase('backward');
        setCurrentStep(null);
        break;

      case 'backward_complete':
        setRunMode('done');
        setCurrentStep(null);
        setAllNodeStatuses('done');
        break;

      case 'error':
        setRunMode('error');
        if (msg.layer_id) setNodeStatus(msg.layer_id, 'error');
        break;

      default:
        break;
    }
  }, []);

  const start = useCallback((graph, config, file) => {
    if (sessionRef.current) sessionRef.current.close();

    const session = createStepSession({
      onOpen: () => {
        session.send('start', {
          graph,
          run_config: config,
          has_file: !!file,
        });
      },
      onMessage: handleMessage,
      onClose: () => {
        setStepSession(null);
        sessionRef.current = null;
      },
      onError: () => setRunMode('error'),
    });

    sessionRef.current = session;
    setStepSession(session);
    setRunMode('connecting');
    setAllNodeStatuses('locked');
  }, [handleMessage]);

  const next = useCallback(() => {
    sessionRef.current?.send('next');
  }, []);

  const prev = useCallback(() => {
    sessionRef.current?.send('prev');
  }, []);

  const followBranch = useCallback((branchId) => {
    sessionRef.current?.send('follow', { branch: branchId });
  }, []);

  const stop = useCallback(() => {
    sessionRef.current?.send('stop');
    sessionRef.current?.close();
    setRunMode('idle');
    setAllNodeStatuses('idle');
  }, []);

  return { start, next, prev, followBranch, stop };
}

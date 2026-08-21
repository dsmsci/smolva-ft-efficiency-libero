# Предсказания и evaluation protocol

P1. Seen checkpoint даст неодинаковый zero-shot success по трём targets; wrong instruction должен снижать success, если language реально влияет на behavior.

P2. Naive target fine-tune улучшится с K и может выйти на ранний ceiling.

P3. H1 (action-conditioned latent dynamics) даст более заметный gain на K=10/25, чем на K=5.

P4. H2 (video progress RA-BC) даст более заметный gain на K=5/10, чем на K=25.

P5. TimeRewarder checkpoint ranking будет положительно связан с simulator success; Robometer может оказаться сильнее как in-domain LIBERO reference.

P6. RA-BC может повысить learned reward без роста simulator success; это будет трактоваться как evidence of proxy gaming, а не автоматически как reward hacking.

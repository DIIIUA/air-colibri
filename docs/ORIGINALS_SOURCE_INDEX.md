# Указатель исходников: две оригинальные топологии

Статический source audit, 13 ноября 2026. [Начать с карты](ORIGINALS_TOPOLOGY_RU.md).

SHA-256 относится ко всему файлу и подтверждает его идентичность, а не полное чтение/исполнение каждой ветки. Диапазоны указывают механизмы, на которые опирается карта.

Локальные checkout origin ведут на форки DIIIUA. Наличие обоих закреплённых commits в авторских репозиториях проверено GitHub API; новые upstream версии не подменяют этот срез.

- **airllm**: [e496ed3c5c8f614df943b9401cf17e156ef7aca0](https://github.com/lyogavin/airllm/commit/e496ed3c5c8f614df943b9401cf17e156ef7aca0); worktree clean.

- **colibri**: [0a5504a1dc64f86fcd0fe7db298b0448c730e7e6](https://github.com/JustVugg/colibri/commit/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6); worktree clean.


## A01 — AutoModel dispatch

`airllm/air_llm/airllm/auto_model.py`

[L17–L79](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/auto_model.py#L17-L79)


SHA-256: `31d159b99900073e84dd0ab40a3b9673c3dbdd6741987e56ac4d57bb23cf3cd7`


## A02 — Constructor, meta model and initialization

`airllm/air_llm/airllm/airllm_base.py`

[L58–L215](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_base.py#L58-L215) · [L267–L379](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_base.py#L267-L379)


SHA-256: `aab819f954876b946f3be73bc7600bfc2c138aefb775510d909321e64798858b`


## A03 — CPU loading, device materialization, resident tables

`airllm/air_llm/airllm/airllm_base.py`

[L430–L458](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_base.py#L430-L458) · [L600–L632](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_base.py#L600-L632) · [L673–L747](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_base.py#L673-L747)


SHA-256: `aab819f954876b946f3be73bc7600bfc2c138aefb775510d909321e64798858b`


## A04 — Layer/expert hooks, prefetch and delegation

`airllm/air_llm/airllm/airllm_base.py`

[L749–L929](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_base.py#L749-L929)


SHA-256: `aab819f954876b946f3be73bc7600bfc2c138aefb775510d909321e64798858b`


## A05 — Cleanup, device choice, mmap tables

`airllm/air_llm/airllm/utils.py`

[L78–L102](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/utils.py#L78-L102) · [L239–L343](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/utils.py#L239-L343)


SHA-256: `37ff2b8b2a40310ab1c1c8dd19658891d555ff8fe74f9066b24a59f507679389`


## A06 — Checkpoint preparation and split lifecycle

`airllm/air_llm/airllm/utils.py`

[L395–L422](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/utils.py#L395-L422) · [L544–L886](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/utils.py#L544-L886)


SHA-256: `37ff2b8b2a40310ab1c1c8dd19658891d555ff8fe74f9066b24a59f507679389`


## A07 — Whole-layer versus subset reader

`airllm/air_llm/airllm/utils.py`

[L425–L460](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/utils.py#L425-L460)


SHA-256: `37ff2b8b2a40310ab1c1c8dd19658891d555ff8fe74f9066b24a59f507679389`


## A08 — Platform persister selection

`airllm/air_llm/airllm/persist/model_persister.py`

[L1–L39](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/persist/model_persister.py#L1-L39)


SHA-256: `9f9dc0f2f38592abe5345e4492d24d5de12d9c433ede57cbafd4db63e734687c`


## A08 — Safetensors persistence and marker

`airllm/air_llm/airllm/persist/safetensor_model_persister.py`

[L1–L43](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/persist/safetensor_model_persister.py#L1-L43)


SHA-256: `acec72660d53f9354949bebf9aab0aa68323db9eb9a74d20473b0c87d20f290f`


## A08 — MLX persistence and name mapping

`airllm/air_llm/airllm/persist/mlx_model_persister.py`

[L1–L113](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/persist/mlx_model_persister.py#L1-L113)


SHA-256: `ca7044a60f38d92627cafea6c93d537dcc68fcac794b16f204caebf934ee0422`


## A09 — Kimi expert specialization

`airllm/air_llm/airllm/airllm_kimi_k3.py`

[L1–L32](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_kimi_k3.py#L1-L32)


SHA-256: `dd69016f41e1aad8884d8e744d67aa3d7f9d8dc8553f624eedf5b452478e4e09`


## A10 — Qwen dense VL and resident vision

`airllm/air_llm/airllm/airllm_qwen3_5.py`

[L1–L30](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_qwen3_5.py#L1-L30)


SHA-256: `f9214d133c1a409c501fb05db8bcc6503d6eef996df474538e21e227c244ef2c`


## A11 — Qwen4 packed experts and PLE

`airllm/air_llm/airllm/airllm_qwen4_exp.py`

[L1–L62](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_qwen4_exp.py#L1-L62)


SHA-256: `922c9d74921cfcab108240e62ec76280c38fdb698ee94a1e389ec23784fc5e14`


## A12 — MLX executor lifecycle

`airllm/air_llm/airllm/airllm_llama_mlx.py`

[L233–L461](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_llama_mlx.py#L233-L461)


SHA-256: `fc48a9efcaa0bc6f47933688a4c951ca6556c11319eb35b8bccc566831a1650c`


## A13 — LoRA reload/recompute lifecycle

`airllm/air_llm/airllm/airllm_lora.py`

[L1–L76](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_lora.py#L1-L76) · [L192–L219](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_lora.py#L192-L219) · [L329–L378](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_lora.py#L329-L378) · [L381–L397](https://github.com/lyogavin/airllm/blob/e496ed3c5c8f614df943b9401cf17e156ef7aca0/air_llm/airllm/airllm_lora.py#L381-L397)


SHA-256: `063a3693561fb6355e3833dd0f85c309605e9869e039a9d22f1fca44ac2c5855`


## C01 — Family dispatch and capability registry

`colibri/c/family_registry.py`

[L28–L240](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/family_registry.py#L28-L240) · [L1303–L1885](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/family_registry.py#L1303-L1885) · [L1953–L2130](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/family_registry.py#L1953-L2130)


SHA-256: `1a0458545e902dff3d9ec61f348c06087af653e364fd8a3757f0617ded8255b9`


## C02 — CLI dispatch and serving entry

`colibri/c/coli`

[L430–L450](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/coli#L430-L450) · [L1758–L1795](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/coli#L1758-L1795) · [L3431–L3491](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/coli#L3431-L3491)


SHA-256: `40693ba8899f2efafc56a24cf276a6c55eb7e4bfcc5b389692f60c6094e79428`


## C03 — Resource plan outputs and environment handoff

`colibri/c/resource_plan.py`

[L3700–L3892](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/resource_plan.py#L3700-L3892)


SHA-256: `401bdbcb7fafce71ee586caaa1718d34aaa184cd3490042776029dbbaec53dc4`


## C03 — Setup discovery and model preparation

`colibri/c/setup_flow.py`

[L1840–L1860](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/setup_flow.py#L1840-L1860) · [L2001–L2196](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/setup_flow.py#L2001-L2196) · [L2367–L2380](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/setup_flow.py#L2367-L2380)


SHA-256: `d64ebc6a4617f73afbcdbc660c7e3c301c46214e2229e715258380806498b891`


## C04 — Native scoring primitives

`colibri/c/tier.h`

[L1–L70](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/tier.h#L1-L70)


SHA-256: `a48e23cd6c814827aef7c440bc1116a2b7e7a55861eed471bcdd1b9323f31931`


## C05 — Gateway engine ownership and shutdown

`colibri/c/openai_server.py`

[L6770–L6915](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/openai_server.py#L6770-L6915) · [L7639–L7680](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/openai_server.py#L7639-L7680) · [L10220–L10270](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/openai_server.py#L10220-L10270) · [L10360–L10397](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/openai_server.py#L10360-L10397)


SHA-256: `3f6c8c199468d43b24c97cc1eac1ab222782e1c3edb6ed1ac6223d63ff0522a5`


## C06 — OLMoE initialization and backend selection

`colibri/c/olmoe.c`

[L900–L1062](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L900-L1062)


SHA-256: `ed1da72ab3149e830526ce878d00a9a8e543f232b19ec2b4776bff7af82cf829`


## C06 — Native shard reads and mapping boundaries

`colibri/c/st.h`

[L1340–L1555](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/st.h#L1340-L1555)


SHA-256: `cbfc7518b871b6b228d7160da00bbc5b4003a5d50a267afc335dba70a889d94b`


## C07 — OLMoE slot ownership, loading and KV-room shrink

`colibri/c/olmoe.c`

[L280–L307](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L280-L307) · [L1067–L1175](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L1067-L1175) · [L1358–L1586](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L1358-L1586)


SHA-256: `ed1da72ab3149e830526ce878d00a9a8e543f232b19ec2b4776bff7af82cf829`


## C08 — OLMoE route, grouped expert compute and layer loop

`colibri/c/olmoe.c`

[L1657–L1705](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L1657-L1705) · [L1790–L1860](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L1790-L1860) · [L1959–L1991](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L1959-L1991) · [L2149–L2240](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L2149-L2240)


SHA-256: `ed1da72ab3149e830526ce878d00a9a8e543f232b19ec2b4776bff7af82cf829`


## C09 — OLMoE speculative worker and process lifetime

`colibri/c/olmoe.c`

[L2291–L2460](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L2291-L2460) · [L3342–L3420](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L3342-L3420) · [L3470–L3559](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L3470-L3559)


SHA-256: `ed1da72ab3149e830526ce878d00a9a8e543f232b19ec2b4776bff7af82cf829`


## C10 — Common device tier: staging, quiesce, heat and shutdown

`colibri/c/vk_tier.c`

[L35–L52](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/vk_tier.c#L35-L52) · [L549–L611](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/vk_tier.c#L549-L611) · [L730–L824](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/vk_tier.c#L730-L824) · [L2220–L2248](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/vk_tier.c#L2220-L2248) · [L2385–L2479](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/vk_tier.c#L2385-L2479) · [L3268–L3285](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/vk_tier.c#L3268-L3285) · [L3322–L3349](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/vk_tier.c#L3322-L3349)


SHA-256: `52d28ac10c0077e28329d31cb340144014fa8031746ae66f3fd38e1590370ddc`


## C10 — OLMoE tier source ownership and RAM callback

`colibri/c/olmoe.c`

[L1178–L1260](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L1178-L1260)


SHA-256: `ed1da72ab3149e830526ce878d00a9a8e543f232b19ec2b4776bff7af82cf829`


## C11 — OLMoE full-resident CUDA chain

`colibri/c/olmoe_cuda_chain.h`

[L1–L62](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe_cuda_chain.h#L1-L62) · [L118–L143](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe_cuda_chain.h#L118-L143) · [L217–L342](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe_cuda_chain.h#L217-L342) · [L401–L448](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe_cuda_chain.h#L401-L448) · [L524–L606](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe_cuda_chain.h#L524-L606)


SHA-256: `c87ffb825d384a32c8f9dc1ad4ae3c8031f4d79632edee55364b78d620d2a54c`


## C12 — CUDA frame synchronization and shutdown

`colibri/c/cuda_chain.cu`

[L1573–L1586](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/cuda_chain.cu#L1573-L1586) · [L1667–L1685](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/cuda_chain.cu#L1667-L1685) · [L2440–L2501](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/cuda_chain.cu#L2440-L2501)


SHA-256: `0e4dee42f3314b5867a0d29597049ce103dcd0c5487d1dbfa541dc5569919784`


## C13 — CUDA tier interface and limitations

`colibri/c/cu_tier.h`

[L1–L65](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/cu_tier.h#L1-L65)


SHA-256: `16ce9c6e6183629889e5aa9a02331d0399c058e5da04b2e46e310a0ae4e3162a`


## C13 — Compile shared tier over CUDA ops

`colibri/c/cu_tier.c`

[L1–L145](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/cu_tier.c#L1-L145)


SHA-256: `cdd4745a6ba0696c28558b1faf11089c7f978dc24d72520d77783491ccf27d8f`


## C14 — Qwen hybrid layers, dispatch, MTP boundary, Dense residency

`colibri/c/qwen36.c`

[L5956–L5995](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/qwen36.c#L5956-L5995) · [L6495–L6547](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/qwen36.c#L6495-L6547) · [L6595–L6605](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/qwen36.c#L6595-L6605) · [L6688–L6803](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/qwen36.c#L6688-L6803) · [L9620–L9688](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/qwen36.c#L9620-L9688)


SHA-256: `1ec64bf5c0e24e4cdfcf7756df78f276f5d434161954a546ded17793921a825e`


## C15 — GLM per-layer repin scope and emitted-token cadence

`colibri/c/colibri.c`

[L10889–L10905](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/colibri.c#L10889-L10905) · [L10980–L10995](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/colibri.c#L10980-L10995) · [L11057–L11067](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/colibri.c#L11057-L11067) · [L14355–L14366](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/colibri.c#L14355-L14366)


SHA-256: `0faa40ad16a621248d78a6da9f68b4b218a21d15b223735dea69eb60688bee23`


## C16 — Documented optional route substitution versus prefetch

`colibri/docs/CACHE_ROUTE.md`

[L1–L89](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/docs/CACHE_ROUTE.md#L1-L89)


SHA-256: `fe82a88610e596b538f83119fc8554132cdb662dd312d7b8b423473e1f51a2ba`


## C17 — OLMoE session/prefix, serving and embedded cleanup

`colibri/c/olmoe.c`

[L2694–L2835](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L2694-L2835) · [L2957–L3022](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L2957-L3022) · [L3121–L3278](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L3121-L3278) · [L3576–L3605](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L3576-L3605) · [L3680–L3743](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/olmoe.c#L3680-L3743)


SHA-256: `ed1da72ab3149e830526ce878d00a9a8e543f232b19ec2b4776bff7af82cf829`


## C18 — Documented Metal topology and its numerical boundaries

`colibri/docs/metal.md`

[L1–L55](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/docs/metal.md#L1-L55)


SHA-256: `57443372f1a163e18d0417e075913a6e8f85f768b7b9356a5fe98c5c238ab000`


## C18 — Qwen Metal affine-store initialization

`colibri/c/qwen36.c`

[L9620–L9637](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/qwen36.c#L9620-L9637)


SHA-256: `1ec64bf5c0e24e4cdfcf7756df78f276f5d434161954a546ded17793921a825e`


## C19 — Documented expert-worker topology

`colibri/docs/cluster.md`

[L1–L65](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/docs/cluster.md#L1-L65)


SHA-256: `16d55d51ccaa278ecda865d72d752225b47423809f1267e6692e6339d6283b7f`


## C20 — Image encode/denoise/decode boundary

`colibri/c/qwenimage.c`

[L1059–L1105](https://github.com/JustVugg/colibri/blob/0a5504a1dc64f86fcd0fe7db298b0448c730e7e6/c/qwenimage.c#L1059-L1105)


SHA-256: `74b57779f20591bc28f61b3cf3377e8db5525818b42fae07b60d19ba57e6256b`


## I01 — Bridge, local events and decay cadence

`aircolibri/src/upstream_bridge.py`

[src/upstream_bridge.py](../src/upstream_bridge.py) (working-tree snapshot)


SHA-256: `420f935a64be5180bbb440ebd724a211f79541c02d01a3ed275003c1ec536263`


## I01 — C ABI bridge to upstream tier.h

`aircolibri/src/native/tier_bridge.c`

[src/native/tier_bridge.c](../src/native/tier_bridge.c) (working-tree snapshot)


SHA-256: `a63c164c417ed8bed9a93f8f19c4ba261a0d8b0cef1a9127513e5dcefeb2874b`


## I02 — Current OLMoE HF integration

`aircolibri/src/olmoe_integrated.py`

[src/olmoe_integrated.py](../src/olmoe_integrated.py) (working-tree snapshot)


SHA-256: `c19f582feb944e96572c56a0d7ad69a22d81f62897b050bad87616bb632f9235`


## I03 — Shared arenas and GPU residency

`aircolibri/src/shared_resources.py`

[src/shared_resources.py](../src/shared_resources.py) (working-tree snapshot)


SHA-256: `0d8a8379e09a1130108dfea9f91006695e3e99934d15a1fa848b01950aec9d2f`


## I04 — Existing project Ring and fences

`aircolibri/src/gpu_ring_pipeline.py`

[src/gpu_ring_pipeline.py](../src/gpu_ring_pipeline.py) (working-tree snapshot)


SHA-256: `1a4827a937ede7ccf8c57d522584d7fbc1f82b472ae535490b6da7303a10bbc6`


## I05 — Current startup resource estimates

`aircolibri/src/resource_plan.py`

[src/resource_plan.py](../src/resource_plan.py) (working-tree snapshot)


SHA-256: `a9c3850f8d676361beaf4803346c1164330933b7fbe90f160f38e7f0ff943476`


## Пределы покрытия

Полная карта lifecycle не равна аудиту каждого kernel. Внешняя HF-математика, остальные family-specific native kernels, все formats/backends, MTP verification и cluster recovery требуют собственных проверок. Список всех 13 dispatch entries и границ — в [JSON](evidence/2026-11-13/original-topology.json).

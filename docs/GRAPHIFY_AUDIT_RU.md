# Дополнительная сверка топологий через Graphify

13 ноября 2026. Установленный `graphifyy 0.9.18`; локальный режим `extract --code-only
--no-cluster --max-workers 2`. Анализ выполнен отдельно для двух исходных checkout-ов.
Веса не загружались, исходники не переносились в новую папку. Графы и cache анализа записаны
в `C:\air-colibri\data\shared-moe-20261113\topology-graphify`.

| Оригинал | Обнаружено code files | Узлы | Связи | LLM tokens |
|---|---:|---:|---:|---:|
| AirLLM | 58 | 715 | 1 467 | 0 |
| Colibrì | 1 241 | 24 210 | 69 599 | 0 |

Счётчик code files принадлежит классификатору Graphify: он включает некоторые JSON/config
файлы. Граф включает tests, tooling и условные backend branches, а не только active inference.
У 2 входных файлов AirLLM и 93 файлов Colibrì получилось ноль узлов; предупреждения сохранены.
Поэтому эти числа — объём извлечённого материала, не процент полноты понимания программы.

## Проверенные связи

- AirLLM: `_pre_hook → _load_streamed_layer → load_layer_subset` найден как двухшаговый
  directed path. Source уточняет guard: subset используется для слоя с expert specialization.
- Colibrì: `step → kv_room_fit`, `step → layers_forward_range`, `step → occ_forward`;
  это разные участки/условные ветки, не три одновременно выбранных executors.
- Colibrì: `expert_get → cache_reserve → read/publish` представлен вызовами
  `cache_reserve`, `load_expert_merged`, `cache_publish` из `expert_get`. Порядок и lock scope
  подтверждены исходным C, а не порядком edges в JSON.
- `occ_start` связан с geometry/room checks, загрузкой expert bytes и созданием device table;
  исходник подтверждает full-resident характер этого OLMoE CUDA пути.

## Найденные ограничения автоматического графа

1. **Конкретная ложная ссылка:** `c_olmoe_expert_get → c_tests_test_qwen36_tier_hold_sleep_ms`.
   Вызов в `olmoe.c:1412` использует macro `sleep_ms`, определённый в строках 87–91 через
   `Sleep/usleep`. Тестовая функция не является его runtime target. Raw edge оставлен в
   исходном графе и помечен недостоверным в компактном evidence.
2. **Declarations вместо реализации:** nodes `vkt_note`, `vkt_join`, `vkt_shutdown` указывают
   на `vk_tier.h`, хотя важное поведение находится в `vk_tier.c`. Название узла не заменяет
   сверку implementation и conditional compilation.
3. **Несколько отношений между одной парой:** диагностика обнаружила 25 групп AirLLM и
   143 группы Colibrì, где simple directed graph потерял бы часть relations. Поэтому raw
   relations сохранены; clustering/объединение по совпавшим именам для вывода эквивалентности
   не применялось.
4. **Внешние endpoints:** 283 edges AirLLM и 5 519 Colibrì имеют endpoints вне таблицы nodes.
   Часть относится к external dependencies. Это не означает, что все такие edges ошибочны,
   но их нельзя считать полностью разрешённым executable call graph.

## Как использованы результаты

На [наложенной карте](FUNCTIONAL_OVERLAY_RU.md) функции сопоставлены **по обязанности**,
с ручной сверкой shapes, residency, guards, clocks и lifecycle. Машиночитаемый crosswalk хранит
точные Graphify node IDs, но его отношения явно названы `analyst_functional_analogy_not_call_edge`.
Никаких выдуманных calls между двумя оригинальными программами не добавлено.

Публичное компактное evidence: [graphify-topology.json](evidence/2026-11-13/graphify-topology.json).
Исходные графы остаются локально, отдельно:

- `data/shared-moe-20261113/topology-graphify/airllm/graphify-out/graph.json`
- `data/shared-moe-20261113/topology-graphify/colibri/graphify-out/graph.json`

Для продолжения доступны `graphify path`, `graphify explain` и raw JSON с source locations.
Примеры и диагностика привязаны к закреплённым revisions в [указателе](ORIGINALS_SOURCE_INDEX.md).

# PURRGE — Состояние на 8 апреля 2026

## Что работает
- Sparse HashMap grid 1024³ (7.8M voxels, ~30 MB vs 8 GB flat)
- 3 комнаты: гостиная 500×600, кухня 400×600, спальня 980×350 (все в вокселях = см)
- FurnitureObj: отдельные объекты с массой, physics, support check
- Мебель мелкая (32 вокс grid = 32 см) — НУЖНО МАСШТАБИРОВАТЬ
- Кот: WASD+AD повороты, мышь-камера, LMB scratch, jump, sprint
- Собака: sleep/patrol/block, 26-костный скелет
- HUD: полоски без подписей
- Маркер наведения: мышь → raycast → оранжевый крестик
- Частицы при scratch
- Chunk meshing: 128³ chunks, ~165 occupied, 14.4M tris

## Что НЕ работает / не сделано
- Мебель 32 вокселя = 32 см. Нужно 200+ для дивана.
- Нет коллизий кота с мебелью (проходит насквозь)
- Нет стартового экрана / меню / настроек
- HUD без подписей
- Нет звука
- Нет редактора кота
- Нет цепных реакций (connectivity check отключён — работает только furniture support)
- Скорость кота 55 вокс/с = слишком медленно для 1024 grid

## Ключевые файлы
- src/main.rs — game loop, ~1100 строк
- src/apartment.rs — sparse grid + generator
- src/furniture.rs — FurnitureObj + catalogue
- src/render.rs — chunk renderer + HUD overlay
- src/core/ — engine (skeleton, body, meshing, physics, materials)
- src/game/ — meters, dog AI, scoring, timer, physics

## Номер билда
Build 13 "Tom" — обновлять в 3 местах:
1. src/main.rs строка 2 (comment)
2. src/main.rs .with_title(...)
3. src/main.rs println в main()
4. src/main.rs set_title в HUD update

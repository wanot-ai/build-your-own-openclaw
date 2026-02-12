# NanoClaw Project Notes

## Architecture
- Single Node.js process
- WhatsApp via baileys
- SQLite for all data
- Agents in Apple Container / Docker

## Key Files
- src/index.ts — main loop
- src/container-runner.ts — agent execution
- src/db.ts — database
- src/group-queue.ts — concurrency

## TODO
- [ ] Add Telegram channel
- [ ] Implement vector search for memory
- [x] Per-group CLAUDE.md

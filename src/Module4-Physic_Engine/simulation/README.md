# Module 4 simulation – Physics Engine Integration & Topology Guard

Module 4 simulation lấy đầu ra của Module 4 preprocessing (`src/Module4-Physic_Engine/preprocessing/Output/<dataset>/`): visual mesh và collision hull của từng segment, mesh đã sửa topology, cùng `roi_config`. Từ đó nó dựng **Unified Physics Scene** trong MuJoCo và đo chất lượng mesh qua mô phỏng.

Chạy từ gốc repo, trong env có `mujoco`, `open3d`, `scipy`, `trimesh`:

```powershell
python scripts/06_run_simulation.py                      # viewer: double-click vật 1 rồi vật 2 -> vật 1 tự tới va chạm vật 2 (tự build nếu chưa có scene)
python scripts/06_run_simulation.py --auto               # viewer: agent tự chọn vật gần nhất ngay khi mở
python scripts/06_run_simulation.py --no-view            # chạy nhiệm vụ không cửa sổ (agent -> vật gần nhất), chỉ ghi báo cáo
python scripts/06_run_simulation.py --no-view --actor=ROI_39 --target=ROI_11   # vật 1 -> vật 2, không cửa sổ
python scripts/06_run_simulation.py --start=1.0,-1.0,0.5 # đặt điểm xuất phát của agent (m)
python scripts/06_run_simulation.py --manual             # viewer cũ: kéo thả vật bằng chuột
python scripts/06_run_simulation.py --rebuild --check    # build lại + kiểm tra ổn định 4 s + viewer
python scripts/06_run_simulation.py --benchmark          # so sánh M_High / M_Adaptive / M_Coarse
```

Các cờ:

| Cờ | Ý nghĩa |
|---|---|
| `--rebuild` | Build lại scene |
| `--no-bg-collision` | Background không có collision |
| `--check` | Chạy thử 4 s không cửa sổ |
| `--no-view` | Không mở viewer: nhiệm vụ chạy headless, chỉ ghi `task_report.json` |
| `--start=x,y,z` | Điểm xuất phát của agent (mặc định: giữa phòng, cao 1.2 m so với ROI thấp nhất) |
| `--actor=ROI_x` | Vật này (thay cho agent) di chuyển tới va chạm mục tiêu |
| `--target=ROI_y` | Mục tiêu, thay cho vật gần nhất |
| `--auto` | Viewer: không chờ double-click, agent tự đi tới vật gần nhất |
| `--manual` | Không chạy nhiệm vụ: viewer để kéo thả vật bằng tay như trước |
| `--variant=unified\|high\|adaptive\|coarse` | Chọn phiên bản mesh |
| `--input=office0\|room0\|...` | Bộ dữ liệu (tên folder trong `preprocessing/Input/`), mặc định `office0` |
| `--no-sleep` | Tắt chế độ đứng yên: mọi vật chạy vật lý ngay từ t=0, ở vị trí đã được resolver nhấc/đẩy |

## Nhiệm vụ mô phỏng tự động (`auto_task.py`)

Mô phỏng ở đây là một hệ thống giả lập có tính chất vật lý tương đương cảnh thật. Mục đích là phân tích và dự đoán kết quả của một tương tác trước khi làm thật. Vì vậy không còn kéo thả thủ công nữa: một agent (hộp kinematic 12 cm, đóng vai đầu robot) tự thực hiện nhiệm vụ.

1. **Chọn vật di chuyển (actor) và mục tiêu:**
   - Trong viewer: **double-click (chuột trái, 2 lần nhanh) vật 1** (actor), rồi **double-click vật 2** (mục tiêu). Vật 1 tự đi tới và va chạm vào vật 2. Click 1 lần không chọn được vật.
   - Mỗi lần chọn, terminal in `[Viewer] selected ROI_x`. Nếu không thấy dòng này thì click chưa được nhận.
   - Cách khác: gõ vào terminal đang chạy viewer `ROI_39 ROI_41` (hoặc `39 41`) rồi Enter.
   - Vật 1 phải là vật dynamic: vật static (tường, sofa, tủ, ...) bị từ chối vì đã gắn cố định vào world.
   - Vật 2 có thể là vật bất kỳ, kể cả tường (khi đó kết luận là `IMPACT`).
   - Double-click một vật khác bất kỳ lúc nào sẽ chọn lại actor: lượt đang chạy bị huỷ.
   - Actor được kéo bằng **lực vật lý** (bộ PD + bù trọng lực qua `xfrc_applied`, giới hạn 200 N), nên khối lượng và collision hull của chính nó tham gia vào va chạm.
   - Khi chạy `--no-view` hoặc `--auto` mà không có `--actor`, actor là agent (hộp kinematic 12 cm) và mục tiêu là vật dynamic gần nhất. Segment `unknown` chỉ được chọn khi không còn vật nào khác.
2. **Quỹ đạo thẳng:** vật 1 đi theo đường thẳng từ vị trí hiện tại tới vật 2.
   - Điểm nhắm là độ cao đầu tiên trong hộp của vật 2 (gần tâm nhất) mà tia MuJoCo từ vật 1 trúng chính vật 2. Ví dụ bàn thì nhắm vào mặt bàn, không lọt giữa các chân.
   - Không có tầm nhìn thẳng tới vật 2 thì vẫn đi thẳng, nhưng sẽ cảnh báo là có thể đụng vật khác trước.
3. **Di chuyển:**
   - Đi 0.6 m/s tới sát mục tiêu (khoảng cách lấy từ tia chạm bề mặt), sau đó chậm lại 0.25 m/s.
   - Actor là vật thể thì giữ cách sàn scan 6 cm. Nếu bị vướng (gờ thảm, chỗ lồi của scan) thì nâng thêm 5 cm, tối đa 4 lần.
4. **Va chạm:** tiếp tục đi thẳng tới khi chạm, rồi đẩy thêm 20 cm theo cùng hướng.
   - Agent dừng sớm nếu lực trung bình trong 50 ms vượt 200 N, vì robot thật có giới hạn lực.
   - Actor là vật thể thì kết luận `BLOCKED` khi nó ép ở mức lực trần 200 N quá 0.5 s.
   - Xong lượt đẩy: agent lùi thẳng về chỗ cũ, còn actor là vật thể thì được thả ra để rơi/đứng tự do.
5. **Quan sát và báo cáo:** sau khi lùi về / thả ra, task chờ tới khi mọi vật đứng yên (tối đa 3 s) rồi ghi `task_report.json`.

| Kết luận | Ý nghĩa |
|---|---|
| `PUSHED` | Vật trượt đi và vẫn đứng |
| `TOPPLED` | Vật bị đổ (nghiêng > 45°) |
| `BLOCKED` | Vật không dịch chuyển, hoặc cần lực lớn hơn 200 N |
| `IMPACT` | Mục tiêu là vật static (tường, sofa, ...): chỉ đo lực va chạm và phản ứng của actor |
| `MISSED` | Actor không chạm được mục tiêu; báo cáo ghi vật đã chặn nó (`blocked_by`) |

Báo cáo gồm:
- thời điểm, vị trí và lực của va chạm (peak / mean / xung lực);
- độ dịch chuyển, độ nghiêng và góc xoay của vật;
- các vật khác bị kéo theo (hiệu ứng dây chuyền) và vật agent đụng phải trên đường;
- số lần "nổ" Topology Guard ghi nhận.

Báo cáo còn ghi actor kết thúc ở đâu và có bị đổ không. Trong viewer, Backspace đưa mọi thứ về đầu và chờ chọn lại.

**Chế độ đứng yên tới khi bị tác động (mặc định cho viewer và `--check`):**
- Mọi vật dynamic nằm **đúng vị trí scan** và đứng yên.
- Một vật chỉ "thức dậy" khi bị kéo bằng chuột, hoặc khi có vật đang chuyển động hay probe chạm vào nó.
- Vật vừa thức bị giới hạn vận tốc trong 0.5 s đầu, để các chỗ lún có sẵn từ lúc scan tách ra từ từ.
- Backspace đưa mọi vật về vị trí scan.
- Benchmark không dùng chế độ này.

**Đường nối giữa các segment:** tam giác nằm vắt qua ranh giới giữa hai segment không thuộc riêng segment nào. Chúng được giữ trong background, chỉ dùng để hiển thị và không có collision, để mesh không bị hở viền.

Biến môi trường:

| Biến | Ý nghĩa |
|---|---|
| `MODULE6_TARGET_ROI` | Vật mục tiêu của benchmark. Mặc định là segment dynamic đầu tiên có nhãn, tức ghế `ROI_4` |
| `MODULE5_STATIC_IDS` / `MODULE5_DYNAMIC_IDS` | Ép một segment thành static / dynamic |
| `MODULE5_COLLISION_METHOD` | `patch` (mặc định) hoặc `coacd` |

Output nằm ở `src/Module4-Physic_Engine/simulation/Output/<dataset>/`:
- `scenes/<variant>/Unified_MuJoCo.xml`
- `scenes/<variant>/penetration_report.json`
- `benchmark_report.json`
- `task_report.json` (nhiệm vụ mô phỏng tự động)

## Các file

| File | Vai trò (README Module 4 simulation, mục 6.3) |
|---|---|
| `env_mujoco.py` | B1–B3, B5–B8. Dựng scene:<br>• background ngoài segment được tách theo màu `.ply`, collision hull dạng tấm;<br>• segment static được gắn cố định vào world, segment dynamic có `freejoint`;<br>• khối lượng và quán tính ước lượng từ AABB;<br>• khử xuyên lấn ban đầu (đẩy tối đa 5 cm, quá mức thì `exclude` cặp va chạm);<br>• drop test tìm đồ gắn tường/trần;<br>• viewer. |
| `auto_task.py` | Nhiệm vụ mô phỏng tự động:<br>• chọn actor + mục tiêu bằng double-click (hoặc agent → vật gần nhất);<br>• actor là vật thể được kéo bằng lực PD;<br>• quỹ đạo thẳng, điểm nhắm bằng ray cast;<br>• state machine `WAIT → SETTLE → APPROACH → PUSH → HOLD → RETREAT → OBSERVE → DONE`;<br>• phân tích kết quả va chạm và ghi `task_report.json`. |
| `topology_guard.py` | • Kiểm tra asset trước khi nạp.<br>• State machine `IDLE → VALIDATING → SETTLING → RUNNING → FAULT`: khi gặp NaN, `BADQACC` hoặc vận tốc vượt ngưỡng thì khôi phục snapshot an toàn gần nhất và đếm là một lần "nổ".<br>• Viewer, `--check` và benchmark đều chạy qua guard này. |
| `collision_verifier.py` | B7, B9.<br>• Mỗi contact: điểm, pháp tuyến, độ lún, lực.<br>• Interpenetration (max / mean / p95).<br>• FP = va chạm "ma" so với M_High; FN = va chạm bị bỏ sót.<br>• Error_ROI = Chamfer giữa các điểm contact probe–ROI của variant và của M_High. |
| `benchmarks.py` | B4, B10, B11.<br>• Cùng một quỹ đạo probe cho cả 3 variant.<br>• Metric: polygon, FPS, thời gian, số lần nổ, độ ổn định contact, FP/FN, Error_ROI so với τ_ROI, Reduction. |

## Các phiên bản mesh (B10)

| Variant | Segment mục tiêu | Segment khác | Background |
|---|---|---|---|
| `unified` (viewer) | full-res | full-res | QEM ≤ 20k tam giác |
| `high` = M_High | full-res | full-res | full-res |
| `adaptive` = M_Adaptive | full-res | QEM × `target_ratio` (0.3) | QEM × 0.3 |
| `coarse` = M_Coarse | QEM × 0.3 | QEM × 0.3 | QEM × 0.3 |

Với các segment bị QEM, collision hull được sinh lại từ mesh đã QEM (cùng hàm `build_hull_parts` của Module 4 preprocessing). Cả 3 variant dùng chung một tập vật static/dynamic, lấy theo kết quả của `high`, để kết quả so sánh được với nhau.

## Nhiệm vụ benchmark (B4)

Probe là một mocap body hình hộp cạnh 12 cm, đóng vai đầu robot. Nó xuất phát cách vật mục tiêu khoảng 0.6 m, về phía tâm phòng. Timeline:

| Thời điểm | Hoạt động |
|---|---|
| 0–1 s | Đứng yên chờ scene ổn định |
| 1–3 s | Đẩy thẳng tới centroid của vật |
| 3–4 s | Lùi về |
| Dừng ở 4.5 s | Kết thúc |

Quỹ đạo giống hệt nhau cho mọi variant.

## Công thức

- `Reduction = 1 − N_variant / N_High`, tính riêng cho visual, collision và tổng.
- `collision_accuracy = 1 − (FP + FN) / (số contact tham chiếu + FP)`.
- `Error_ROI ≤ τ_ROI`, với τ lấy từ `TAU_ROI` trong `roi_config.py` (4.5 mm).

## Giới hạn

- **Vật bị kẹt trong sàn scan:** ở `room0`, hai ghế đẩu ROI_39 / ROI_41 không nhúc nhích kể cả khi agent đẩy với lực 15 kN. Collision hull của sàn (background) giữ chúng lại, nên kết quả luôn là `BLOCKED`; cần sửa collision của background ở Module 4 preprocessing. Tương tự, `environment` (phần scan chưa segment) có thể chặn đường actor.
- **Chưa có robot thật:** agent là probe kinematic.
  - Actor đi theo đường thẳng, không né vật cản: vật nằm trên đường đi sẽ bị đụng hoặc chặn actor lại (ghi trong `blocked_by`).
  - Lực đẩy được giới hạn bằng cách dừng lại khi vượt ngưỡng, không phải bằng một bộ điều khiển lực.
  - Probe không va chạm với vật static, vì MuJoCo không sinh contact giữa mocap và world, nên tường không chặn được nó.
- **M_Coarse được tạo ngay trong Module 4 simulation** bằng QEM toàn cục trên từng mesh, chứ không phải một output riêng của Module 4 preprocessing.
- **Một số cặp vật bị tắt va chạm:** mesh quét là mặt hở, nên một số cặp vẫn xuyên nhau quá 5 cm sau khi resolver đẩy. Các cặp này bị `exclude` và được ghi trong `penetration_report.json`.
- **`topology_guard` chỉ phục hồi được, không sửa được mesh:** nó khôi phục trạng thái khi mô phỏng nổ. Mesh lỗi phải được sửa ở Module 4 preprocessing.


Both `--input=room0` and `--input room0` are supported. Dataset inputs are under `preprocessing/Input/<dataset>/` and outputs are separated by dataset.

import queue
import threading
import time
import dobotArm
import lib.DobotDllType as dType

class RobotController:
    """
    Runs Dobot commands in a background thread so the camera loop never blocks.
    A maxsize-1 queue means only one pending command waits at a time.
    """
    def __init__(self, api):
        self.api            = api
        self.busy           = threading.Event()
        self._q             = queue.Queue(maxsize=1)
        self.gripper_closed = False
        self._t             = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        
        self.SPEED_NORMAL = 50
        self.SPEED_SLOW = 25
        self.speed_is_slow = False

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            action, args = item
            self.busy.set()
            try:
                if action == "xyz":
                    dobotArm.move_to_xyz(self.api, *args)
                elif action == "close":
                    dobotArm.close_gripper(self.api)
                    self.gripper_closed = True
                elif action == "open":
                    dobotArm.open_gripper(self.api)
                    dobotArm.stop_pump(self.api)
                    self.gripper_closed = False
                elif action == "home":
                    dobotArm.move_to_home(self.api)
                    dobotArm.open_gripper(self.api)
                    dobotArm.stop_pump(self.api)
                    self.gripper_closed = False
                elif action == "batch":
                    self._execute_batch(*args)
            except Exception as exc:
                print(f"[robot] {action} failed: {exc}")
            finally:
                self.busy.clear()

    def send(self, action, *args, drop_if_busy=False):
        if drop_if_busy and self.busy.is_set():
            return
        cmd = (action, args)
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(cmd)
        except queue.Full:
            pass

    def stop(self):
        self._q.put(None)
        
    def set_speed(self, slow=False):
        """Immediately change the speed of the robot. (isQueued=0 bypasses the queue)"""
        if slow and not self.speed_is_slow:
            dType.SetPTPCommonParams(self.api, self.SPEED_SLOW, self.SPEED_SLOW, isQueued=0)
            self.speed_is_slow = True
            print("Hand detected — speed reduced to 50%.")
        elif not slow and self.speed_is_slow:
            dType.SetPTPCommonParams(self.api, self.SPEED_NORMAL, self.SPEED_NORMAL, isQueued=0)
            self.speed_is_slow = False
            print("Hand cleared — speed restored to normal.")

    def _execute_batch(self, pick_list, drop_list, z_safe, z_pick):
        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)
        time.sleep(0.5)

        batch_size = min(len(pick_list), len(drop_list))
        print(f"\n[PHASE 3] Executing batch of {batch_size} operations.")

        for i in range(batch_size):
            pick_x, pick_y = pick_list[i]
            drop_x, drop_y = drop_list[i]
            print(f"Task {i+1}: Moving {pick_x, pick_y} to {drop_x, drop_y}")

            dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_safe)
            time.sleep(0.6)
            dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_pick)
            time.sleep(0.6)

            dobotArm.close_gripper(self.api)
            time.sleep(0.6)
            dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_safe)
            time.sleep(0.6)

            dobotArm.move_to_xyz(self.api, drop_x, drop_y, z_safe)
            dobotArm.open_gripper(self.api)
            dobotArm.stop_pump(self.api)
            time.sleep(0.4)
            dobotArm.move_to_xyz(self.api, drop_x, drop_y, z_safe)

        if len(pick_list) > len(drop_list):
            drop_x, drop_y = drop_list[0]
            for i in range(batch_size, len(pick_list)):
                pick_x, pick_y = pick_list[i]
                dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_safe)
                dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_pick)
                time.sleep(0.2)
                dobotArm.close_gripper(self.api)
                time.sleep(0.5)
                dobotArm.move_to_xyz(self.api, pick_x, pick_y, z_safe)
                time.sleep(0.1)

                dobotArm.move_to_xyz(self.api, drop_x, drop_y, z_safe)
                dobotArm.open_gripper(self.api)
                dobotArm.stop_pump(self.api)
                dobotArm.move_to_xyz(self.api, drop_x, drop_y, z_safe)

        print("\nBatch Complete.")

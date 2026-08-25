# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler


def make_scheduler(main_result=True, draft_result=True):
    scheduler = KVCacheV2Scheduler.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.try_allocate_generation.return_value = main_result
    scheduler.draft_kv_cache_manager = MagicMock()
    scheduler.draft_kv_cache_manager.try_allocate_generation.return_value = draft_result
    return scheduler


def test_generation_allocation_reserves_both_kv_caches():
    scheduler = make_scheduler()
    request = MagicMock()

    assert scheduler._try_allocate_generation(request)
    scheduler.kv_cache_manager.try_allocate_generation.assert_called_once_with(request)
    scheduler.draft_kv_cache_manager.try_allocate_generation.assert_called_once_with(request)


def test_generation_allocation_rolls_back_main_when_draft_is_full():
    scheduler = make_scheduler(draft_result=False)
    request = MagicMock()

    assert not scheduler._try_allocate_generation(request)
    scheduler.kv_cache_manager.revert_allocate_generation.assert_called_once_with(request)


def test_generation_allocation_stops_when_main_is_full():
    scheduler = make_scheduler(main_result=False)
    request = MagicMock()

    assert not scheduler._try_allocate_generation(request)
    scheduler.draft_kv_cache_manager.try_allocate_generation.assert_not_called()
    scheduler.kv_cache_manager.revert_allocate_generation.assert_not_called()


def test_completion_pending_is_not_a_deadlock():
    assert not KVCacheV2Scheduler._is_generation_deadlock(2, 13, [], [], set())


def test_exhausted_generation_without_progress_is_a_deadlock():
    assert KVCacheV2Scheduler._is_generation_deadlock(2, 0, [], [], set())

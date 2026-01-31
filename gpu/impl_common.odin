
#+private
package gpu

import vmem "core:mem/virtual"
import "core:mem"
import "core:sync"
import "core:log"
import "base:runtime"

Resource_Block_Size :: 256

// Implementation of a thread-safe resource pool to be used for no_gfx_api handles
Resource_Pool :: struct($Handle_T: typeid, $Info_T: typeid)
{
    array: [dynamic]Resource(Info_T),
    free_list: [dynamic]u32,
    lock: sync.Atomic_Mutex,
    init: bool,
}

Resource :: struct($T: typeid)
{
    info: T,
    gen: u32,
}

Resource_Key :: struct
{
    idx: u32,
    gen: u32,
}

pool_init :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T))
{
    init = true

    // Reserve element 0
    pool_add(pool, Info_T {})
}

pool_get :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), handle: Handle_T) -> Info_T
{
    assert(init)

    sync.guard(&lock)

    key := transmute(Resource_Key) handle
    assert(key.gen == array[key.idx].gen)
    return array[key.idx].info
}

pool_set :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), handle: Handle_T, new_info: Info_T)
{
    assert(init)

    sync.guard(&lock)

    key := transmute(Resource_Key) handle
    assert(key.gen == array[key.idx].gen)
    array[key.idx].info = new_info
}

pool_add :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), info: Info_T) -> Handle_T
{
    assert(init)

    sync.guard(&lock)

    free_idx: u32
    if len(free_list) > 0 {
        free_idx = pop(&free_list)
    } else {
        append(&array, Resource(Info_T) {})
        free_idx = u32(len(array)) - 1
    }

    gen := array[free_idx].gen
    array[free_idx].info = info

    key := Resource_Key { idx = free_idx, gen = gen }
    return transmute(Handle_T) key
}

pool_remove :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), handle: Handle_T)
{
    assert(init)

    sync.guard(&lock)

    key := transmute(Resource_Key) handle

    el := &array[key.idx]

    if key.gen != el.gen {
        return
    }

    el.gen += 1
    if key.idx == u32(len(array)) {
        pop(&array)
    } else {
        append(&free_list, key.idx)
    }
}

pool_destroy :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T))
{
    assert(init)

    delete(array)
    delete(free_list)
    array = {}
    free_list = {}
}

// Scratch arena implementation
@(deferred_out = release_scratch)
acquire_scratch :: proc(used_allocators: ..mem.Allocator) -> (mem.Allocator, vmem.Arena_Temp)
{
    @(thread_local) scratch_arenas: [4]vmem.Arena = {}
    @(thread_local) initialized: bool = false
    if !initialized
    {
        for &scratch in scratch_arenas
        {
            error := vmem.arena_init_growing(&scratch)
            assert(error == nil)
        }

        initialized = true
    }

    available_arena: ^vmem.Arena
    if len(used_allocators) < 1
    {
        available_arena = &scratch_arenas[0]
    }
    else
    {
        for &scratch in scratch_arenas
        {
            for used_alloc in used_allocators
            {
                // NOTE: We assume that if the data points to the same exact address,
                // it's an arena allocator and it's the same arena
                if used_alloc.data != &scratch
                {
                    available_arena = &scratch
                    break
                }

                if available_arena != nil do break
            }
        }
    }

    assert(available_arena != nil, "Available scratch arena not found.")

    return vmem.arena_allocator(available_arena), vmem.arena_temp_begin(available_arena)
}

release_scratch :: #force_inline proc(allocator: mem.Allocator, temp: vmem.Arena_Temp)
{
    vmem.arena_temp_end(temp)
}

// Utilities
get_mip_dimensions_u32 :: proc(texture_dimensions: [3]u32, mip_level: u32) -> [3]u32
{
    return {
        max(1, u32(f32(texture_dimensions.x) / f32(u32(1) << mip_level))),
        max(1, u32(f32(texture_dimensions.y) / f32(u32(1) << mip_level))),
        max(1, u32(f32(texture_dimensions.z) / f32(u32(1) << mip_level))),
    }
}

get_mip_dimensions_i32 :: proc(texture_dimensions: [3]i32, mip_level: u32) -> [3]i32
{
    return {
        max(1, i32(f32(texture_dimensions.x) / f32(i32(1) << mip_level))),
        max(1, i32(f32(texture_dimensions.y) / f32(i32(1) << mip_level))),
        max(1, i32(f32(texture_dimensions.z) / f32(i32(1) << mip_level))),
    }
}

get_mip_dimensions :: proc { get_mip_dimensions_u32, get_mip_dimensions_i32 }

align_up :: proc(x, align: u64) -> (aligned: u64)
{
    assert(0 == (align & (align - 1)), "must align to a power of two")
    return (x + (align - 1)) &~ (align - 1)
}

// Misc

fatal_error :: proc(fmt: string, args: ..any, location := #caller_location)
{
    log.fatalf(fmt, ..args, location = location)
    runtime.panic("")
}

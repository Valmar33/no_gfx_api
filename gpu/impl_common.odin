
#+private
package gpu

import vmem "core:mem/virtual"
import "core:mem"
import "core:sync"
import "core:log"
import "base:runtime"
import intr "base:intrinsics"

// Implementation of a thread-safe resource pool to be used for no_gfx_api handles
Resource_Block_Size :: 256
Resource_Block :: struct($T: typeid)
{
    res: [Resource_Block_Size]Resource(T),
}

Resource_Pool :: struct($Handle_T: typeid, $Info_T: typeid) where size_of(Handle_T) == 8
{
    blocks: [dynamic]^Resource_Block(Info_T),  // Makes pointers to elements stable.
    res_count: u32,  // Number of resources added to the "blocks" data structure.
    freelist: [dynamic]u32,
    lock: sync.Atomic_Mutex,
    init: bool,
}

Resource :: struct($T: typeid)
{
    info: T,
    gen: u32,
    lock: sync.Benaphore,
    alive: bool,
    meta: Resource_Metadata,
}

Resource_Metadata :: struct
{
    name: string,
    created_at: runtime.Source_Code_Location,
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
    pool_add(pool, Info_T {}, {})
}

pool_check :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), handle: Handle_T, name: string, loc: runtime.Source_Code_Location) -> bool
{
    assert(init)

    if handle == nil {
        log.errorf("'%v' handle is nil.", name, location = loc)
        return false
    }

    key := transmute(Resource_Key) handle

    block_idx, el_idx := pool_get_idx(key.idx)

    el := intr.volatile_load(&blocks[block_idx].res[el_idx])
    if key.gen != el.gen {
        log.error("'%v' handle is used after it has been freed, or it's corrupted in some other way.", name, location = loc)
        return false
    }

    return true
}

pool_check_no_message :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), handle: Handle_T, name: string, loc: runtime.Source_Code_Location) -> bool
{
    assert(init)

    if handle == nil {
        return false
    }

    key := transmute(Resource_Key) handle

    block_idx, el_idx := pool_get_idx(key.idx)

    el := intr.volatile_load(&blocks[block_idx].res[el_idx])
    if key.gen != el.gen {
        return false
    }

    return true
}

pool_get_alive_list :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), arena: runtime.Allocator) -> []Resource(Info_T)
{
    res := make([dynamic]Resource(Info_T), allocator = arena)
    for i in 1..<res_count
    {
        block_idx, el_idx := pool_get_idx(i)
        el := intr.volatile_load(&blocks[block_idx].res[el_idx])
        if el.alive do append(&res, el)
    }

    return res[:]
}

pool_get :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), handle: Handle_T) -> Info_T
{
    assert(init)
    assert(handle != {})
    key := transmute(Resource_Key) handle

    block_idx, el_idx := pool_get_idx(key.idx)

    el := intr.volatile_load(&blocks[block_idx].res[el_idx])
    assert(key.gen == el.gen)
    return el.info
}

// To be used like this:
// if resource, lock := pool_get_mut(&pool, handle); sync.guard(lock)
pool_get_mut :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), handle: Handle_T) -> (^Info_T, ^sync.Benaphore)
{
    assert(init)
    assert(handle != {})
    key := transmute(Resource_Key) handle

    block_idx, el_idx := pool_get_idx(key.idx)

    el := &blocks[block_idx].res[el_idx]
    el_gen := intr.volatile_load(&el.gen)
    assert(key.gen == el_gen)
    return &el.info, &blocks[block_idx].res[el_idx].lock
}

pool_get_lock :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), handle: Handle_T) -> ^sync.Benaphore
{
    assert(init)
    assert(handle != {})
    key := transmute(Resource_Key) handle

    block_idx, el_idx := pool_get_idx(key.idx)

    el := &blocks[block_idx].res[el_idx]
    el_gen := intr.volatile_load(&el.gen)
    assert(key.gen == el_gen)
    return &blocks[block_idx].res[el_idx].lock
}

pool_add :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), info: Info_T, meta: Resource_Metadata) -> Handle_T
{
    assert(init)
    sync.guard(&lock)

    free_idx: u32
    if len(freelist) > 0 {
        free_idx = pop(&freelist)
    } else {
        pool_append_internal(pool, Resource(Info_T) {})
        free_idx = res_count - 1
    }

    block_idx, el_idx := pool_get_idx(free_idx)
    blocks[block_idx].res[el_idx].info = info
    gen := blocks[block_idx].res[el_idx].gen
    blocks[block_idx].res[el_idx].alive = true
    blocks[block_idx].res[el_idx].meta = meta

    key := Resource_Key { idx = free_idx, gen = gen }
    return transmute(Handle_T) key
}

pool_remove :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T), handle: Handle_T)
{
    assert(init)
    assert(handle != {})
    sync.guard(&lock)

    key := transmute(Resource_Key) handle

    block_idx, el_idx := pool_get_idx(key.idx)
    el := &blocks[block_idx].res[el_idx]
    el.alive = false
    assert(key.gen == el.gen)

    el.gen += 1
    append(&freelist, key.idx)
}

pool_destroy :: proc(using pool: ^Resource_Pool($Handle_T, $Info_T))
{
    assert(init)
    sync.guard(&lock)

    for block in blocks {
        free(block)
    }
    delete(blocks)
    blocks = {}
    freelist = nil
    init = false
}

@(private="file")
pool_get_idx :: #force_inline proc(idx: u32) -> (block_idx: u32, el_idx: u32)
{
    return idx / Resource_Block_Size, idx % Resource_Block_Size
}

@(private="file")
pool_append_internal :: #force_inline proc(using pool: ^Resource_Pool($Handle_T, $Info_T), el: Resource(Info_T))
{
    do_alloc_new_block := res_count % Resource_Block_Size == 0
    res_count += 1
    if do_alloc_new_block {
        new_block := new(Resource_Block(Info_T))
        append(&blocks, new_block)
    }

    block_idx, el_idx := pool_get_idx(res_count - 1)
    blocks[block_idx].res[el_idx] = el
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

is_block_compressed :: #force_inline proc(format: Texture_Format) -> bool
{
    #partial switch format
    {
        case .BC1_RGBA_Unorm,
             .BC3_RGBA_Unorm,
             .BC7_RGBA_Unorm,
             .ASTC_4x4_RGBA_Unorm,
             .ETC2_RGB8_Unorm,
             .ETC2_RGBA8_Unorm,
             .EAC_R11_Unorm,
             .EAC_RG11_Unorm:
            return true
    }
    return false
}

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

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/gqa_attention_fixture.h"

#include <cstring>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::gqa;

namespace {

int archive_roundtrip() {
    LayoutBuilder builder;
    auto layout = plan_paged_kv_pool(builder, {.page_group_count = 80,
        .logical_page_capacity = 200, .table_rows = 1,
        .planes = {{DType::I8, 256, 2}, {DType::I8, 256, 2},
                   {DType::FP16, 4, 2}, {DType::FP16, 4, 2}}, .ram_enabled = true});
    DeviceBuffer storage(builder.finish(256));
    PagedKVPool pool({storage.p, storage.bytes}, layout);
    auto allocation = pool.reserve(80);
    allocation.bind_row(0);
    const auto payload = [](int page, std::size_t plane, std::size_t bytes) {
        std::vector<unsigned char> value(bytes);
        for (std::size_t i = 0; i < bytes; ++i) { value[i] = (page * 17 + plane * 37 + i * 7) % 251; }
        return value;
    };
    for (int end = 8; end <= 200; end += 8) {
        allocation.materialize_pages(end);
        for (int p = end - 8; p < end; ++p) {
            const auto physical = allocation.page_ids()[p];
            if (physical < 0) { throw std::runtime_error("append page was evicted"); }
            for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
                auto bytes = payload(p, plane, pool.plane(plane).nb[3]);
                CUDA_CHECK(cudaMemcpy(static_cast<char*>(pool.plane(plane).data) +
                    physical * bytes.size(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
            }
        }
    }
    const int resident_sink = allocation.page_ids()[0];
    std::vector<int> ranking{20, 21, 22, 23, 24, 25, 26, 27};
    allocation.prefer_pages(ranking);
    allocation.materialize_pages(200);
    CUDA_CHECK(cudaDeviceSynchronize());
    int failures = allocation.page_ids()[0] != resident_sink;
    for (int p : ranking) {
        const int physical = allocation.page_ids()[p];
        if (physical < 0) { ++failures; continue; }
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            auto expected = payload(p, plane, pool.plane(plane).nb[3]);
            std::vector<unsigned char> actual(expected.size());
            CUDA_CHECK(cudaMemcpy(actual.data(), static_cast<char*>(pool.plane(plane).data) +
                physical * actual.size(), actual.size(), cudaMemcpyDeviceToHost));
            if (actual != expected) { ++failures; }
        }
    }
    // Retained-prefix entitlement resize must not truncate logical history to resident count.
    PagedKVResize resize{&allocation, allocation.mapped_page_count(), 80};
    resize_paged_kv_bundle(std::span(&resize, 1));
    failures += allocation.mapped_token_capacity() != 200 * 64;
    allocation.trim_tokens(150 * 64 - 3);
    allocation.materialize_tokens(150 * 64 - 3);
    failures += allocation.mapped_token_capacity() != 150 * 64;
    failures += allocation.mapped_page_count() > 80;
    failures += allocation.ram_transfer_bytes() == 0;
    allocation.release();
    failures += pool.free_pages() != 80 || pool.entitled_pages() != 0;
    std::cout << "RAM exact code/scale archive roundtrip: " << failures << " failures\n";
    return failures;
}

int sparse_attention(DType dtype, int tokens, bool cached) {
    const Geometry geometry{"tp2", 12, 2};
    const int base = 1024;
    HostCache host = make_cache(geometry, dtype, base + tokens, 100);
    const auto q = make_bf16_values(256 * 12 * tokens, 20, -0.25f, 0.25f);
    const auto k = make_bf16_values(256 * 2 * tokens, 21, -0.25f, 0.25f);
    const auto v = make_bf16_values(256 * 2 * tokens, 22, -1.0f, 1.0f);
    std::vector<int> positions(tokens);
    for (int t = 0; t < tokens; ++t) { positions[t] = base + t; }
    DeviceCache device_cache(host, MappingPattern::Fragmented);
    if (!cached) { append_cache(host, k, v, positions); }
    const std::vector<int> selected{0, 3, 15, 16};
    const auto reference = ideal_attention(q, host, positions, selected);
    auto single = device_cache.view();
    std::vector<int> original(single.block_table.ne[0]);
    CUDA_CHECK(cudaMemcpy(original.data(), single.block_table.data, original.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int> indices(original.size(), -1), table(original.size(), original.back());
    for (std::size_t i = 0; i < selected.size(); ++i) {
        indices[selected[i]] = i;
        table[i] = original[selected[i]];
    }
    GuardedDeviceBuffer di(indices.size() * 4), dt(table.size() * 4);
    di.copy_from_host(indices.data(), indices.size() * 4);
    dt.copy_from_host(table.data(), table.size() * 4);
    single.read_indices = Tensor(di.data(), DType::I32, {int(indices.size())});
    single.read_table = Tensor(dt.data(), DType::I32, {int(table.size())});
    single.read_capacity = selected.size() * 64;
    const auto qb = to_bf16_bits(q), kb = to_bf16_bits(k), vb = to_bf16_bits(v);
    GuardedDeviceBuffer dq(qb.size() * 2), dk(kb.size() * 2), dv(vb.size() * 2),
        dp(positions.size() * 4), dout(qb.size() * 2), drow(4);
    dq.copy_from_host(qb.data(), qb.size() * 2);
    dk.copy_from_host(kb.data(), kb.size() * 2);
    dv.copy_from_host(vb.data(), vb.size() * 2);
    dp.copy_from_host(positions.data(), positions.size() * 4);
    const int row = 0;
    drow.copy_from_host(&row, 4);
    Tensor tq(dq.data(), DType::BF16, {256, 12, tokens}),
        tk(dk.data(), DType::BF16, {256, 2, tokens}), tv(dv.data(), DType::BF16, {256, 2, tokens}),
        tp(dp.data(), DType::I32, {tokens}), out(dout.data(), DType::BF16, {256, 12, tokens}),
        tr(drow.data(), DType::I32, {1});
    const auto capacity = ops::gqa_attention_workspace_capacity_bytes(12, dtype,
        {1, static_cast<unsigned>(base + tokens)}, 1, tokens, tokens, true);
    GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena arena({scratch.data(), capacity});
    if (cached) {
        ops::gqa_attention_cached(tq, tp, kAttentionScale, single,
            {1, static_cast<unsigned>(base + tokens)}, arena, out, nullptr);
    } else {
        auto batch = device_cache.batch_view();
        batch.read_indices = single.read_indices;
        batch.read_tables = single.read_table.view({int(table.size()), 1});
        batch.read_capacity = single.read_capacity;
        ops::gqa_attention(tq, tk, tv, tp, Tensor{}, tr, kAttentionScale, batch,
            {static_cast<unsigned>(base + tokens), static_cast<unsigned>(base + tokens)}, arena, out, nullptr);
    }
    cuda_synchronize();
    const std::string label = std::string("sparse original-position oracle ") + cache_name(dtype) +
        " T=" + std::to_string(tokens) + (cached ? " cached" : " append");
    int failures = verify_attention(label, bf16_bits_to_double(copy_from_guarded<std::uint16_t>(dout, qb.size())),
                                     reference, attention_criterion(dtype));
    if (!cached) { failures += verify_cache(label, device_cache.snapshot(), host); }
    failures += scratch.verify_guards(label.c_str());
    failures += verify_positions(label, dp, positions);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    int failures = archive_roundtrip();
    for (auto dtype : {DType::BF16, DType::I8}) {
        failures += sparse_attention(dtype, 4, false);
        failures += sparse_attention(dtype, 4, true);
        failures += sparse_attention(dtype, 64, false);
    }
    return failures == 0 ? 0 : 1;
}

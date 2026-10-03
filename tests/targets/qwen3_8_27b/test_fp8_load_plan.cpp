#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include <cstdlib>
#include <iostream>

int main() {
    const char* path=std::getenv("NINFER_QWEN38_FP8_ARTIFACT");
    if(!path || !*path) {std::cout<<"SKIP: select a real native FP8 artifact\n";return 77;}
    using namespace ninfer;
    namespace target=targets::qwen3_6_27b::detail;
    try {
        artifact::Reader reader(path);
        if(reader.identity().model_id!="qwen3.8-27b" || reader.identity().weights_id!="fp8") {
            throw std::runtime_error("not the registered native FP8 identity");
        }
        for(const auto speculative:{SpeculativeBackend::None,SpeculativeBackend::Mtp}) {
            for(const auto proposal:{ProposalHead::Full,ProposalHead::Optimized}) {
                artifact::Binder binder(reader,4);
                const auto plan=target::bind_artifact(binder,target::WeightsProfile::Qwen38Fp8,
                    {.speculative=speculative,.proposal_head=proposal},4);
                if(plan.materialization.device_count!=4 || plan.materialization.host_objects.size()!=6 ||
                   plan.bindings.mtp_stem_format!=artifact::NumericFormat::BF16 ||
                   plan.bindings.mtp_format!=artifact::NumericFormat::FP8_E4M3FN_BLOCK128_BF16S ||
                   plan.bindings.output_head.format!=artifact::NumericFormat::BF16) {
                    throw std::runtime_error("incomplete native FP8 startup roles");
                }
                for(int rank=1;rank<4;++rank) {
                    if(plan.materialization.device_capacity_bytes[rank]!=
                       plan.materialization.device_capacity_bytes[0]) {
                        throw std::runtime_error("native FP8 four-rank placement is unbalanced");
                    }
                }
                std::cout<<"native FP8 TP4 load plan spec="<<static_cast<int>(speculative)
                         <<" proposal="<<static_cast<int>(proposal)<<" bytes/rank="
                         <<plan.materialization.device_capacity_bytes[0]<<'\n';
            }
        }
        for(const int tp:{1,2}) {
            artifact::Binder binder(reader,tp);
            bool rejected=false;
            try {(void)target::bind_artifact(binder,target::WeightsProfile::Qwen38Fp8,{},tp);}
            catch(const std::invalid_argument&) {rejected=true;}
            if(!rejected) {throw std::runtime_error("unqualified native FP8 width was admitted");}
        }
        std::cout<<"OK native FP8 real artifact binding; no GPU allocation\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

"""Acceptance is separate from evaluation; ordinary adoption stays manual."""


class Acceptance:
    def manual(self, versions, candidate, reason, *, lineage=None):
        return versions.adopt_manual(candidate, reason, lineage=lineage)

    def experiment(self, versions, candidate, enabled, *, lineage=None):
        return versions.adopt_auto(candidate, lineage=lineage) if enabled else {
            'decision':'pending', 'mode':'manual', 'reason':'Experiment automatic adoption is disabled',
            'baseline_version':versions.active(lineage), 'candidate_version':candidate,
            'scope':{'kind':'campaign' if lineage else 'global','id':lineage,'ref':versions.active_ref(lineage)}}

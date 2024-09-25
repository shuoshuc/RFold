#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from job import TopoType, Job

PHILLY_TRACE = 'data/philly_trace.csv'

class TraceReplay:
    '''
    Replays a given trace file.
    '''
    def __init__(self, tracefile):
        self.jobs = []
        self._parse_trace(tracefile)
        self.job_iter = iter(self.jobs)

    def _parse_trace(self, tracefile):
        '''
        Parses a trace in csv format.
        '''
        with open(tracefile, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # Skips the comment line.
                if line.startswith('#'):
                    continue
                jid, arrival_time_sec, topo_type, shape, job_size, duration_min = line.strip().split(',')
                self.jobs.append(Job(uuid=int(jid), topology=TopoType[topo_type],
                                     shape=tuple(map(int, shape.split('+'))),
                                     size=int(job_size), arrival_time_sec=float(arrival_time_sec),
                                     duration_minutes=float(duration_min)))
    
    def run(self, num_jobs=1):
        '''
        Returns a list of given number of jobs from the trace, in order.
        Each job is used just once, no recycle.
        If the jobs run out, the return list is filled with None to match the requested number.
        '''
        jobs = []
        for _ in range(num_jobs):
            jobs.append(next(self.job_iter, None))
        return jobs

def main():
    trace = TraceReplay(PHILLY_TRACE)
    jobs = trace.run(10)
    for job in jobs:
        print(job)

if __name__ == "__main__":
    main()
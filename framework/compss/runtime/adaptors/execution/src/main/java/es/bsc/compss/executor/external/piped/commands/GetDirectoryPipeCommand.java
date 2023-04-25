/*
 *  Copyright 2002-2022 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package es.bsc.compss.executor.external.piped.commands;

import es.bsc.compss.executor.external.commands.GetDirectoryExternalCommand;


public class GetDirectoryPipeCommand extends GetDirectoryExternalCommand implements PipeCommand {

    /**
     * Constructs an OpenFileCommand out of the message received through the pipe.
     *
     * @param args message received through the pipe
     */
    public GetDirectoryPipeCommand(String[] args) {
        super();
        this.directory = args[2];
    }

    @Override
    public int compareTo(PipeCommand t) {
        int val = Integer.compare(this.getType().ordinal(), t.getType().ordinal());
        if (val == 0) {
            val = this.getDirectory().compareTo(((GetDirectoryExternalCommand) t).getDirectory());
        }
        return val;
    }

    @Override
    public void join(PipeCommand receivedCommand) {
        // Do nothing
    }

}

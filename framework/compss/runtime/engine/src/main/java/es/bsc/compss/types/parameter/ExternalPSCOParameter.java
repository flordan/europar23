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
package es.bsc.compss.types.parameter;

import es.bsc.compss.api.ParameterMonitor;
import es.bsc.compss.types.annotations.parameter.DataType;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.StdIOStream;


public class ExternalPSCOParameter extends DependencyParameter {

    /**
     * Serializable objects Version UID are 1L in all Runtime.
     */
    private static final long serialVersionUID = 1L;

    private final int hashCode;
    private String pscoId;


    /**
     * Creates a new Stream Parameter.
     * 
     * @param direction Parameter direction.
     * @param stream Standard IO Stream flags.
     * @param prefix Parameter prefix.
     * @param name Parameter name.
     * @param weight Parameter weight.
     * @param pscoId Parameter PSCO Id.
     * @param hashCode Parameter object hashcode.
     * @param monitor object to notify to changes on the parameter
     */
    public ExternalPSCOParameter(Direction direction, StdIOStream stream, String prefix, String name, double weight,
        String pscoId, int hashCode, ParameterMonitor monitor) {

        super(DataType.EXTERNAL_PSCO_T, direction, stream, prefix, name, "null", weight, false, monitor);
        this.pscoId = pscoId;
        this.hashCode = hashCode;
    }

    @Override
    public boolean isCollective() {
        return false;
    }

    public String getId() {
        return this.pscoId;
    }

    public void setId(String pscoId) {
        this.pscoId = pscoId;
    }

    public int getCode() {
        return this.hashCode;
    }

    @Override
    public String toString() {
        return "ExternalObjectParameter with Id " + this.pscoId + " and HashCode " + this.hashCode;
    }

    @Override
    public String generateDataTargetName(String tgtName) {
        return getId();
    }

}
